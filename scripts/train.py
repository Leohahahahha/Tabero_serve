import dataclasses
import functools
import json
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.numerics as numerics
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.models.pi0 import Pi0
from openpi.shared.tactile_type import TactileType


def eval_step(rng, state, batch):
    """No augmentation or updates. Fixed RNGs make before/after flow losses comparable."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    if isinstance(model, Pi0):
        loss, components = model.compute_loss(rng, observation, actions, train=False, return_components=True)
        return {"loss": jnp.mean(loss), "action_loss": components["action_loss"]}
    loss = model.compute_loss(rng, observation, actions, train=False)
    return {"loss": jnp.mean(loss)}


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if isinstance(model, Pi0) and getattr(model, "tactile_type", None) is TactileType.EXPERT_HIS_C_FUT:
            # Loss component computation and logging behavior.
            # Tactile/force stream configuration and loss behavior.
            chunked_loss, components = model.compute_loss(
                rng,
                observation,
                actions,
                train=True,
                return_components=True,
            )
            action_loss_mean = components["action_loss"]
            tactile_loss_mean = components["tactile_loss"]
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            action_loss_mean = jnp.array(0.0, dtype=jnp.float32)
            tactile_loss_mean = jnp.array(0.0, dtype=jnp.float32)

        total_loss = jnp.mean(chunked_loss)
        aux = {
            "action_loss": action_loss_mean,
            "tactile_loss": tactile_loss_mean,
        }
        return total_loss, aux

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    finite_checks = {
        "inputs_finite": numerics.all_finite(batch),
        "loss_finite": numerics.all_finite((loss, aux)),
        "grads_finite": numerics.all_finite(grads),
        "updates_finite": numerics.all_finite(updates),
        "candidate_params_finite": numerics.all_finite(new_params),
        "optimizer_finite": numerics.all_finite(new_opt_state),
    }

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )
        finite_checks["ema_finite"] = numerics.all_finite(new_state.ema_params)

    update_applied = numerics.all_finite((loss, aux)) & jnp.all(jnp.stack(list(finite_checks.values())))
    new_state = numerics.accept_finite_update(state, new_state, update_applied)

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        # Implementation note.
        "action_loss": aux["action_loss"],
        "tactile_loss": aux["tactile_loss"],
        **finite_checks,
        "update_applied": update_applied,
        "first_bad_grad_index": numerics.first_nonfinite_leaf(grads),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    eval_loader = None
    if config.eval_interval:
        if config.eval_interval < 0 or config.eval_num_batches <= 0:
            raise ValueError("Evaluation interval and batch count must be positive")
        eval_loader = _data_loader.create_data_loader(
            config, sharding=data_sharding, split="validation", shuffle=False,
            num_batches=config.eval_num_batches,
        )

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    trainable = train_state.params.filter(config.trainable_filter)
    trainable_count = sum(x.size for x in jax.tree.leaves(trainable))
    total_count = sum(x.size for x in jax.tree.leaves(train_state.params))
    logging.info("Trainable parameters: %d / %d (%.3f%%)",
                 trainable_count, total_count, 100 * trainable_count / total_count)
    logging.info("Trainable paths: %s", list(trainable.flat_state()))
    gradient_paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(trainable)[0]]
    if not bool(jax.device_get(numerics.all_finite((train_state.params, train_state.opt_state)))):
        raise FloatingPointError("Initial model or optimizer contains non-finite values")
    if getattr(config.model, "tactile_prefix_lora_rank", 0):
        tactile_lora = {
            k: v for k, v in traverse_util.flatten_dict(trainable.to_pure_dict(), sep="/").items()
            if k.startswith("tactile_prefix_encoder/") and "/lora_" in k
        }
        if not tactile_lora:
            raise ValueError("Tactile LoRA configured but no tactile adapter parameters are trainable")
        logging.info("Trainable tactile LoRA: %d parameters in %d leaves",
                     sum(v.size for v in tactile_lora.values()), len(tactile_lora))
    metrics_path = config.checkpoint_dir / "metrics.jsonl"
    (config.checkpoint_dir / "run_config.txt").write_text(repr(config))

    def save_numerical_failure(info, failed_batch):
        """The compiled step has already rejected the update, preserving a healthy state."""
        info = {k: float(v) for k, v in jax.device_get(info).items()}
        bad_index = int(info["first_bad_grad_index"])
        report = {
            "healthy_updates": int(train_state.step),
            "attempted_update": int(train_state.step) + 1,
            "first_bad_gradient": gradient_paths[bad_index] if bad_index >= 0 else None,
            "metrics": {k: v if np.isfinite(v) else str(v) for k, v in info.items()},
            "seed": config.seed,
            "note": "Invalid update rejected; no batch skipped. State below is BEFORE failed update.",
        }
        obs, targets = failed_batch
        arrays = traverse_util.flatten_dict({"observation": obs.to_dict(), "actions": targets}, sep="/")
        np.savez_compressed(config.checkpoint_dir / "failed_batch.npz",
                            **{k: np.asarray(jax.device_get(v)) for k, v in arrays.items()})
        (config.checkpoint_dir / "numerical_failure.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        healthy_step = max(0, int(train_state.step) - 1)
        if healthy_step not in checkpoint_manager.all_steps():
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, healthy_step)
        checkpoint_manager.wait_until_finished()
        raise FloatingPointError(f"Rejected non-finite update; saved healthy checkpoint and diagnostics: {report}")

    def record_metrics(kind, step, values):
        values = {k: float(v) for k, v in values.items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError(f"Non-finite {kind} metrics at step {step}: {values}")
        with metrics_path.open("a") as f:
            f.write(json.dumps({"kind": kind, "step": step, **values}) + "\n")

    peval_step = jax.jit(eval_step)

    def evaluate(step):
        if eval_loader is None:
            return
        eval_infos = []
        for index, eval_batch in enumerate(eval_loader):
            with sharding.set_mesh(mesh):
                info = peval_step(jax.random.fold_in(jax.random.key(config.seed + 1), index), train_state, eval_batch)
            eval_infos.append(jax.device_get(info))
        if not eval_infos:
            raise ValueError("Validation loader produced no batches")
        means = jax.tree.map(lambda *xs: float(np.mean(xs)), *eval_infos)
        means["num_batches"] = len(eval_infos)
        means["num_frames"] = len(eval_infos) * config.batch_size
        record_metrics("validation", step, means)
        logging.info("Validation step %d: %s", step, means)
        if config.wandb_enabled:
            wandb.log({f"validation/{k}": v for k, v in means.items()}, step=step)

    evaluate(int(train_state.step))

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        # Check every update, before the next batch or averaged logging. Never continue
        # training after rejection: reproduce/fix the failing batch instead.
        if not bool(jax.device_get(info["update_applied"])):
            save_numerical_failure(info, batch)
        infos.append(info)
        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            record_metrics("train", step + 1, reduced_info)
            infos = []
        batch = next(data_iter)

        if eval_loader is not None and ((step + 1) % config.eval_interval == 0 or step == config.num_train_steps - 1):
            evaluate(step + 1)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
