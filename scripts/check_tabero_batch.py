"""Inspect one train and validation batch on CPU; never create a model or update weights."""

import argparse
import dataclasses
import json
import os

# Must precede importing JAX/config. The check does not need a GPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np

from openpi.training import config as configs
from openpi.training import data_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi0_lora_tacfield_local_smoke")
    args = parser.parse_args()
    config = dataclasses.replace(configs.get_config(args.config), num_workers=0)
    for split in ("train", "validation"):
        # Use the same evenly distributed validation sampling as the future training loop.
        loader = data_loader.create_data_loader(
            config,
            split=split,
            num_batches=config.eval_num_batches if split == "validation" else 1,
        )
        observation, actions = next(iter(loader))
        if actions.shape != (config.batch_size, config.model.action_horizon, config.model.action_dim):
            raise ValueError(f"Unexpected action shape: {actions.shape}")
        if observation.tactile_prefix.shape != (config.batch_size, 9, 396):
            raise ValueError(f"Unexpected tactile shape: {observation.tactile_prefix.shape}")
        if observation.tactile_suffix is not None:
            raise ValueError("Action-only tacfield configuration must not load a tactile suffix")
        if np.any(np.asarray(actions)[..., 7:] != 0):
            raise ValueError("Unlabeled action slots must be zero padded")
        summary = {"split": split, "episodes": loader.data_config().episodes, "tensors": {}}
        tree = {"observation": observation.to_dict(), "actions": actions}
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            values = np.asarray(leaf)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {split} tensor: {path}")
            summary["tensors"][jax.tree_util.keystr(path)] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
