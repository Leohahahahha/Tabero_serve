"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
from openpi.shared.tactile_type import TactileType
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType

# Tactile/force stream configuration and loss behavior.
TACTILE_LOSS_WEIGHT: float = 0.1
# Tactile/force stream configuration and loss behavior.
TABERO_TACTILE_HISTORY: int = 8
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class ParameterDtypeRule:
    """Override the storage dtype of trainable parameters whose full path matches a regex."""

    path_regex: str
    dtype: Literal["bfloat16", "float32"]


@dataclasses.dataclass(frozen=True)
class ParameterDtypePolicy:
    """Config-owned storage policy for parameters, gradients and optimizer state."""

    name: str
    default_trainable_dtype: Literal["bfloat16", "float32"]
    overrides: tuple[ParameterDtypeRule, ...] = ()
    gradient_dtype: Literal["match_parameter", "float32"] = "match_parameter"
    optimizer_state_dtype: Literal["match_parameter", "float32"] = "match_parameter"


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Explicit local LeRobot root. Never infer this from a Hugging Face repo id.
    root: str | None = None
    episodes: tuple[int, ...] | None = None
    validation_episodes: tuple[int, ...] = ()
    video_backend: str | None = None
    columns: tuple[str, ...] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Optional Hugging Face git revision for `lerobot.datasets.LeRobotDataset` (branch, tag, or commit).
    # Use when the dataset repo has no `v*` version tag (LeRobot defaults to a semver and calls
    # `get_safe_version`, which requires such a tag). Example: the branch name main.
    lerobot_revision: str | None = None

    rlds_data_dir: str | None = None
    filter_dict_path: str | None = None

    def __post_init__(self):
        for ids in (self.episodes, self.validation_episodes):
            if ids is not None and (len(set(ids)) != len(ids) or any(i < 0 for i in ids)):
                raise ValueError("Episode selectors must be unique non-negative integers")
        if self.episodes is not None and not self.episodes:
            raise ValueError("Training episode selector cannot be empty")
        if self.validation_episodes:
            if self.episodes is None or set(self.episodes) & set(self.validation_episodes):
                raise ValueError("Validation requires explicit, disjoint training episodes")


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # NOTE:
        # Some configs (e.g. SimpleDataConfig) may specify `repo_id` via `base_config=DataConfig(repo_id=...)`
        # instead of setting `DataConfigFactory.repo_id` directly. In that case we must NOT override it with None.
        base = self.base_config or DataConfig()
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else base.repo_id
        asset_id = self.assets.asset_id or base.asset_id or repo_id
        return dataclasses.replace(
            base,
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoTactileDataConfig(DataConfigFactory):
    """
    Tactile/force stream configuration and loss behavior.

    Implementation note.
    Tactile/force stream configuration and loss behavior.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # Implementation note.
                        "observation/gripper_force": "gripper_force",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacImgDataConfig(DataConfigFactory):
    """
    Tactile/force stream configuration and loss behavior.

    Tactile/force stream configuration and loss behavior.
      Implementation note.
    Action/force dimensions and loss handling.
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Implementation note.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.TaberoTacImgInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacFieldDataConfig(DataConfigFactory):
    """
    Tactile/force stream configuration and loss behavior.

    Image stream mapping and masking behavior.
    Tactile/force stream configuration and loss behavior.
      Tactile/force stream configuration and loss behavior.
    """

    extra_delta_transform: bool = True
    use_so3_relative_actions: bool = False
    action_only: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The dedicated action-only adapter uses the real FR3 contract, not a simulation schema.
        input_type = libero_policy.TaberoActionOnlyInputs if self.action_only else libero_policy.TaberoTacFieldInputs
        data_transforms = _transforms.Group(
            inputs=[input_type(model_type=model_config.model_type)],
            outputs=[
                libero_policy.TaberoActionOnlyOutputs() if self.action_only else libero_policy.LiberoForceOutputs()
            ],
        )

        if self.extra_delta_transform:
            # Implementation note.
            # Action/force dimensions and loss handling.
            if self.use_so3_relative_actions:
                data_transforms = data_transforms.push(
                    inputs=[_transforms.RelativePoseActions()],
                    outputs=[_transforms.AbsolutePoseActions()],
                )
            else:
                delta_action_mask = _transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )
        elif self.use_so3_relative_actions:
            raise ValueError("SO(3) relative actions require extra_delta_transform=True")

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacFieldWrenchDataConfig(DataConfigFactory):
    """Marker-motion input with a 13D ``7D action + 6D wrist_wrench`` target."""

    extra_delta_transform: bool = True
    use_so3_relative_actions: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[libero_policy.TaberoActionWrenchInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.TaberoActionWrenchOutputs()],
        )
        if self.extra_delta_transform:
            if self.use_so3_relative_actions:
                data_transforms = data_transforms.push(
                    inputs=[_transforms.RelativePoseActions()],
                    outputs=[_transforms.AbsolutePoseActions()],
                )
            else:
                delta_action_mask = _transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )
        elif self.use_so3_relative_actions:
            raise ValueError("SO(3) relative actions require extra_delta_transform=True")

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacForceDataConfig(DataConfigFactory):
    """
    Tactile/force stream configuration and loss behavior.

    Image stream mapping and masking behavior.
    Tactile/force stream configuration and loss behavior.
      Tactile/force stream configuration and loss behavior.
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[libero_policy.TaberoTacForceInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacForceEncDataConfig(DataConfigFactory):
    """
    Tactile/force stream configuration and loss behavior.

    Implementation note.
    Tactile/force stream configuration and loss behavior.
    Tactile/force stream configuration and loss behavior.
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[libero_policy.TaberoTacForceEncInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoTacAllDataConfig(DataConfigFactory):
    """
    Tactile/force stream configuration and loss behavior.

    Image stream mapping and masking behavior.
        - image                  -> base_0_rgb
        - wrist_image            -> left_wrist_0_rgb
        - tactile_image          -> right_wrist_0_rgb
    Tactile/force stream configuration and loss behavior.
        Tactile/force stream configuration and loss behavior.
          Tactile/force stream configuration and loss behavior.
        Tactile/force stream configuration and loss behavior.
          Tactile/force stream configuration and loss behavior.
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[libero_policy.TaberoTacAllInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoForceOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TaberoNoTactNoForceDataConfig(DataConfigFactory):
    """
    Tactile/force stream configuration and loss behavior.

    Image stream mapping and masking behavior.
    Action/force dimensions and loss handling.
      Loss component computation and logging behavior.
    """

    extra_delta_transform: bool = True
    action_only: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        input_type = (
            libero_policy.TaberoNoTactActionOnlyInputs if self.action_only else libero_policy.TaberoNoTactInputs
        )
        data_transforms = _transforms.Group(
            inputs=[
                # Image stream mapping and masking behavior.
                # Tactile/force stream configuration and loss behavior.
                input_type(model_type=model_config.model_type),
                # Tactile/force stream configuration and loss behavior.
                _transforms.SliceActions(7),
            ],
            outputs=[libero_policy.TaberoActionOnlyOutputs() if self.action_only else libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        base = self.create_base_config(assets_dirs, model_config)
        # Reuse identical train-only state/action statistics, without tactile assets in new checkpoints.
        if self.action_only and base.norm_stats is not None:
            base = dataclasses.replace(
                base, norm_stats={k: v for k, v in base.norm_stats.items() if k in ("state", "actions")}
            )
        return dataclasses.replace(base, data_transforms=data_transforms, model_transforms=model_transforms)


@dataclasses.dataclass(frozen=True)
class TaberoNoTactForceDataConfig(DataConfigFactory):
    """
    Tactile/force stream configuration and loss behavior.

    Implementation note.
    Action/force dimensions and loss handling.
    Implementation note.

    Image stream mapping and masking behavior.
    Implementation note.
    Implementation note.
    """

    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                # Tactile/force stream configuration and loss behavior.
                libero_policy.TaberoNoTactInputs(model_type=model_config.model_type),
            ],
            outputs=[libero_policy.LiberoForceOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoNoTactileDataConfig(DataConfigFactory):
    """Tactile/force stream configuration and loss behavior."""

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Implementation note.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Action/force dimensions and loss handling.
        data_transforms = _transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(model_type=model_config.model_type),
                _transforms.SliceActions(7),
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    # Optional storage-dtype policy for trainable parameters. Optimizer moments inherit these dtypes.
    # None preserves the legacy behavior (trainable parameters remain at their model initialization dtype).
    parameter_dtype_policy: tyro.conf.Suppress[ParameterDtypePolicy | None] = None

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # Disabled for existing configs. Bounded validation samples span the whole held-out dataset.
    eval_interval: int = 0
    eval_num_batches: int = 12
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # If true, block training after every save until Orbax has atomically
    # finalized the checkpoint. The manager still uses its async signal protocol.
    wait_for_checkpoint_on_save: bool = False

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True
    # If true, upload a montage from the first training batch to W&B. Scalar
    # metrics and config metadata are still logged when this is false.
    wandb_log_images: bool = True
    # Legacy configurations name checkpoints with the zero-based loop index.
    # New configurations can instead name them by completed update count.
    checkpoint_step_is_update_count: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_lora_tacimg_tabero",
        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Action/force dimensions and loss handling.
            effective_action_dim=13,
            # Tactile/force stream configuration and loss behavior.
            # Tactile/force stream configuration and loss behavior.
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tactile/force stream configuration and loss behavior.
            # Checkpoint loading behavior for newly added parameters.
            tactile_dim_in=0,
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        data=TaberoTacImgDataConfig(
            # Tactile/force stream configuration and loss behavior.
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                # Implementation note.
                prompt_from_task=True,
            ),
            # Implementation note.
            extra_delta_transform=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        # Checkpoint loading behavior for newly added parameters.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacimgwo_tabero",
        # Tactile/force stream configuration and loss behavior.
        # Action/force dimensions and loss handling.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=0,
            # Implementation note.
            tactile_loss_weight=0.0,
        ),
        data=TaberoTacImgDataConfig(
            repo_id="NathanWu7/tabero_object_25",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacall_tabero",
        # Tactile/force stream configuration and loss behavior.
        #
        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        # Action/force dimensions and loss handling.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Implementation note.
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tactile/force stream configuration and loss behavior.
            # Tactile/force stream configuration and loss behavior.
            # Tactile/force stream configuration and loss behavior.
            tactile_streams=("tactile_suffix", "tactile_prefix"),
            # Tactile/force stream configuration and loss behavior.
            # Tactile/force stream configuration and loss behavior.
            tactile_suffix_placement="prefix",
            # Encoder configuration and sequence handling.
            tactile_dim_in=8 * 6,
            tactile_history=TABERO_TACTILE_HISTORY,
            tactile_encoder_type="mlp",
            tactile_use_reference_frame=False,
            tactile_diff_from_reference=True,
            # Encoder configuration and sequence handling.
            tactile_prefix_dim_in=9 * 198 * 2,
            tactile_prefix_history=TABERO_TACTILE_HISTORY,
            tactile_prefix_encoder_type="tcn",
            tactile_prefix_use_reference_frame=True,
            tactile_prefix_diff_from_reference=False,
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoTacAllDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Tactile/force stream configuration and loss behavior.
        # Implementation note.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_notac_tabero",
        # Implementation note.
        # Image stream mapping and masking behavior.
        #   （TaberoNoTactInputs：right_wrist_0_rgb = 0, image_mask=False）。
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoNoTactNoForceDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_notac_bin_tabero",
        # Dataset format and transform behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoNoTactNoForceDataConfig(
            repo_id="NathanWu7/tabero_binary",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacpred_tabero",
        # Action/force dimensions and loss handling.
        # Image stream mapping and masking behavior.
        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tactile/force stream configuration and loss behavior.
            tactile_dim_in=0,
            tactile_streams=(),
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoNoTactForceDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_lora_tacimg_tabero",
        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            # Checkpoint loading behavior for newly added parameters.
            # Action/force dimensions and loss handling.
            action_expert_variant="gemma_300m_lora",
            discrete_state_input=True,
            # Action/force dimensions and loss handling.
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tactile/force stream configuration and loss behavior.
            tactile_dim_in=0,
            tactile_streams=(),
            tactile_loss_weight=0.01,
        ),
        data=TaberoTacImgDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=1_000_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_lora_tacfield_tabero",
        # Image stream mapping and masking behavior.
        # Tactile/force stream configuration and loss behavior.
        # Action/force dimensions and loss handling.
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            discrete_state_input=True,
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tactile/force stream configuration and loss behavior.
            tactile_dim_in=0,
            tactile_prefix_dim_in=9 * 198 * 2,
            tactile_prefix_history=TABERO_TACTILE_HISTORY,
            tactile_prefix_encoder_type="tcn",
            tactile_prefix_use_reference_frame=True,
            tactile_prefix_diff_from_reference=False,
            tactile_streams=("tactile_prefix",),
            tactile_loss_weight=0.01,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=1_000_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        data=TaberoTacFieldDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Tactile/force stream configuration and loss behavior.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_lora_tacforce_tabero",
        # Tactile/force stream configuration and loss behavior.
        #
        # Encoder configuration and sequence handling.
        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            discrete_state_input=True,
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=0,
            tactile_prefix_dim_in=8 * 6,
            tactile_prefix_history=TABERO_TACTILE_HISTORY,
            tactile_prefix_encoder_type="mlp",
            tactile_prefix_use_reference_frame=False,
            tactile_prefix_diff_from_reference=True,
            tactile_streams=("tactile_prefix",),
            tactile_loss_weight=0.01,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=1_000_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        data=TaberoTacForceEncDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Tactile/force stream configuration and loss behavior.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacfield_tabero",
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Action/force dimensions and loss handling.
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tactile/force stream configuration and loss behavior.
            # Tactile/force stream configuration and loss behavior.
            # Tactile/force stream configuration and loss behavior.
            tactile_dim_in=0,
            tactile_prefix_dim_in=9 * 198 * 2,
            tactile_prefix_history=TABERO_TACTILE_HISTORY,
            tactile_prefix_encoder_type="tcn",
            tactile_prefix_use_reference_frame=True,
            tactile_prefix_diff_from_reference=False,
            # Tactile/force stream configuration and loss behavior.
            tactile_streams=("tactile_prefix",),
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoTacFieldDataConfig(
            repo_id="NathanWu7/tabero_object_25",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            # Tactile/force stream configuration and loss behavior.
            missing_regex=".*",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacfieldwo_tabero",
        # Image stream mapping and masking behavior.
        # Action/force dimensions and loss handling.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=0,
            tactile_prefix_dim_in=9 * 198 * 2,
            tactile_prefix_history=TABERO_TACTILE_HISTORY,
            tactile_prefix_encoder_type="tcn",
            tactile_prefix_use_reference_frame=True,
            tactile_prefix_diff_from_reference=False,
            tactile_streams=("tactile_prefix",),
            # Implementation note.
            tactile_loss_weight=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoTacFieldDataConfig(
            repo_id="NathanWu7/tabero_object_25",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacforce_tabero",
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Action/force dimensions and loss handling.
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tactile/force stream configuration and loss behavior.
            tactile_dim_in=8 * 6,
            # Tactile/force stream configuration and loss behavior.
            tactile_history=TABERO_TACTILE_HISTORY,
            # Tactile/force stream configuration and loss behavior.
            tactile_streams=("tactile_suffix",),
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoTacForceDataConfig(
            repo_id="NathanWu7/tabero_object_25",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacforce_tabero_enc",
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            # Tactile/force stream configuration and loss behavior.
            tactile_dim_in=0,
            tactile_prefix_dim_in=8 * 6,
            tactile_prefix_history=TABERO_TACTILE_HISTORY,
            tactile_prefix_encoder_type="mlp",
            tactile_prefix_use_reference_frame=False,
            tactile_prefix_diff_from_reference=True,
            tactile_streams=("tactile_prefix",),
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoTacForceEncDataConfig(
            repo_id="NathanWu7/tabero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_tacforcewo_tabero",
        # Tactile/force stream configuration and loss behavior.
        # Action/force dimensions and loss handling.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=0,
            tactile_prefix_dim_in=8 * 6,
            tactile_prefix_history=TABERO_TACTILE_HISTORY,
            tactile_prefix_encoder_type="mlp",
            tactile_prefix_use_reference_frame=False,
            tactile_prefix_diff_from_reference=True,
            tactile_streams=("tactile_prefix",),
            # Implementation note.
            tactile_loss_weight=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=TaberoTacForceEncDataConfig(
            repo_id="NathanWu7/tabero_object_25",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*",
        ),
        num_train_steps=50_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_noforce_taforce",
        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Loss component computation and logging behavior.
            # Action/force dimensions and loss handling.
            # Tactile/force stream configuration and loss behavior.
            # Tactile/force stream configuration and loss behavior.
            # Tactile/force stream configuration and loss behavior.
            effective_action_dim=13,
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=0,
            # Tactile/force stream configuration and loss behavior.
            tactile_streams=(),
            tactile_loss_weight=0.0,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=LeRobotLiberoNoTactileDataConfig(
            repo_id="NathanWu7/tabero_force",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_lora_force_taforce",
        # Tactile/force stream configuration and loss behavior.
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Action/force dimensions and loss handling.
            # Action/force dimensions and loss handling.
            # Action/force dimensions and loss handling.
            # Tactile/force stream configuration and loss behavior.
            effective_action_dim=13,
            # Action/force dimensions and loss handling.
            # Implementation note.
            # Reference-frame and history-window handling.
            # Action/force dimensions and loss handling.
            tactile_type=TactileType.EXPERT_HIS_C_FUT,
            tactile_dim=6,
            tactile_dim_in=8 * 6,  # Tactile/force stream configuration and loss behavior.
            tactile_history=TABERO_TACTILE_HISTORY,
            tactile_streams=("tactile_suffix",),
            # Tactile/force stream configuration and loss behavior.
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_lr=2.5e-6,
        ),
        data=LeRobotLiberoTactileDataConfig(
            # Dataset format and transform behavior.
            # Dataset format and transform behavior.
            repo_id="NathanWu7/tabero_force",
            # Implementation note.
            #   ./assets/pi0_libero_force_low_mem_finetune/NathanWu7_tabero_force/...
            # Checkpoint loading behavior for newly added parameters.
            base_config=DataConfig(
                # Dataset format and transform behavior.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Checkpoint loading behavior for newly added parameters.
        # Tactile/force stream configuration and loss behavior.
        # Implementation note.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params",
            missing_regex=".*",
        ),
        # Checkpoint loading behavior for newly added parameters.
        # Implementation note.
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

# Small local action-only continuation; original published configs remain unchanged.
_tabero_pretrained = next(c for c in _CONFIGS if c.name == "pi0_lora_tacfield_tabero")
_CONFIGS.append(
    dataclasses.replace(
        _tabero_pretrained,
        name="pi0_lora_tacfield_local_smoke",
        model=dataclasses.replace(
            _tabero_pretrained.model,
            supervised_action_dim=7,
            tactile_loss_weight=0.0,
            padding_loss_weight=0.0,
        ),
        data=TaberoTacFieldDataConfig(
            repo_id="local/tabero_lerobot_compact_v1",
            base_config=DataConfig(
                root="/data/yanghaojun/datasets/tabero_lerobot_compact_v1",
                # Fixed episode split, interspersed through acquisition order, not random frames.
                episodes=tuple(i for i in range(29) if i not in (4, 14, 24)),
                validation_episodes=(4, 14, 24),
                video_backend="pyav",
                columns=(
                    "state",
                    "actions",
                    "tactile_marker_motion",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ),
                prompt_from_task=True,
            ),
            action_only=True,
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data/yanghaojun/checkpoints/tabero-pretrained/checkpoints/"
            "pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params",
            missing_regex="(?!)",  # No random fallback, including missing LoRA/tactile leaves.
            strict=True,
        ),
        freeze_filter=nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        batch_size=2,
        num_workers=2,
        num_train_steps=100,
        log_interval=10,
        eval_interval=50,
        eval_num_batches=12,
        save_interval=100,
        keep_period=None,
        wandb_enabled=False,
        policy_metadata={
            "robot": "franka_fr3",
            "deployment_target": "real_robot",
            "action_representation": "absolute_xyz_axis_angle_single_finger_m",
            "action_dim": 7,
            "dataset_fps": 10,
            "tactile_input": "rolling_9x198x2_marker_coordinates_left_then_right",
            "tactile_marker_shape": [9, 198, 2],
            "tactile_marker_dtype": "float32",
            "tactile_marker_layout": "reference_then_8_history_frames_left_then_right",
            "predicts_wrench": False,
            "robot_safety_validation_required": True,
        },
        checkpoint_base_dir="/data/yanghaojun/outputs/checkpoints",
        assets_base_dir="/data/yanghaojun/outputs/assets",
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=10, peak_lr=1e-5, decay_steps=100, decay_lr=1e-6),
    )
)

# Sensor-adaptation variant. Only new tactile adapters may be absent from Tabero;
# pretrained tactile kernels/biases and backbone LoRA must still be restored strictly.
_tabero_local_smoke = next(c for c in _CONFIGS if c.name == "pi0_lora_tacfield_local_smoke")
_CONFIGS.append(
    dataclasses.replace(
        _tabero_local_smoke,
        name="pi0_lora_tacfield_local_tactile_lora_smoke",
        model=dataclasses.replace(
            _tabero_local_smoke.model,
            tactile_prefix_lora_rank=16,
            tactile_prefix_lora_alpha=16.0,
        ),
        weight_loader=dataclasses.replace(
            _tabero_local_smoke.weight_loader,
            strict_allow_missing_regex=(
                r"tactile_prefix_encoder/(blocks/block_[01]/kernels/kernel_[012]|"
                r"blocks/block_0/residual_proj|out_proj)/lora_[ab]"
            ),
        ),
        policy_metadata={
            **_tabero_local_smoke.policy_metadata,
            "tactile_adaptation": "tcn_lora",
            "tactile_lora_rank": 16,
            "tactile_lora_alpha": 16.0,
        },
    )
)

# Matched local retraining profiles: start from published Tabero, NOT local smoke99/final2999.
# Only reuse checkpoint2999's train-only normalization assets; no learned weights from that run.
_tabero_comparison_assets = AssetsConfig(
    assets_dir=(
        "/data/yanghaojun/outputs/checkpoints/pi0_lora_tacfield_local_tactile_lora_smoke/"
        "real_fr3_recovery_20260903_112020/2999/assets"
    )
)
_tabero_touch_comparison = dataclasses.replace(
    next(c for c in _CONFIGS if c.name == "pi0_lora_tacfield_local_tactile_lora_smoke"),
    name="pi0_lora_tabero_rgb_state_touch",
    data=dataclasses.replace(_tabero_local_smoke.data, assets=_tabero_comparison_assets),
    batch_size=4,
    num_workers=4,
    num_train_steps=3000,
    eval_interval=250,
    eval_num_batches=173,
    save_interval=100,
    keep_period=500,
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=100, peak_lr=1e-5, decay_steps=3000, decay_lr=1e-6),
)
_CONFIGS.append(_tabero_touch_comparison)

# Fresh v3 real-FR3 tactile run. It restores only the published Tabero weights;
# train-only normalization assets are generated independently for this dataset.
_tabero_v3_touch_20k = dataclasses.replace(
    next(c for c in _CONFIGS if c.name == "pi0_lora_tacfield_local_tactile_lora_smoke"),
    name="pi0_lora_tabero_v3_touch_20k",
    project_name="tabero-vtla",
    data=TaberoTacFieldDataConfig(
        repo_id="local/tabero_lerobot_compact_v3",
        base_config=DataConfig(
            root="/data/yanghaojun/datasets/tabero_lerobot_compact_v3",
            episodes=tuple(i for i in range(29) if i not in (4, 14, 24)),
            validation_episodes=(4, 14, 24),
            video_backend="pyav",
            columns=(
                "state",
                "actions",
                "tactile_marker_motion",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ),
            prompt_from_task=True,
        ),
        action_only=True,
        extra_delta_transform=True,
        use_so3_relative_actions=True,
    ),
    batch_size=4,
    num_workers=4,
    num_train_steps=20_000,
    log_interval=10,
    eval_interval=1_000,
    # v3 held-out episodes contain 690 frames; batch 4 evaluates 688 without
    # changing the global training batch used by the earlier comparisons.
    eval_num_batches=172,
    save_interval=4_000,
    keep_period=4_000,
    wandb_enabled=True,
    # Do not upload camera frames merely because scalar W&B tracking is enabled.
    wandb_log_images=False,
    checkpoint_step_is_update_count=True,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=1e-5,
        decay_steps=20_000,
        decay_lr=1e-6,
    ),
    policy_metadata={
        **_tabero_local_smoke.policy_metadata,
        "dataset_version": "tabero_lerobot_compact_v3",
        "action_label_source": "next_observation_state_within_episode",
        "training_action_representation": "relative_xyz_so3_rotvec_absolute_gripper",
        "terminal_source_frame_omitted": True,
        "dataset_timing_policy": "compact",
        "dataset_missing_candidate_steps": 18,
        "deployment_translation_guard_m_s": 0.02,
        "expert_labels_clipped_to_deployment_rate": False,
        "tactile_adaptation": "tcn_lora",
        "tactile_lora_rank": 16,
        "tactile_lora_alpha": 16.0,
        "task_prompt": (
            "Align the black circular component with the receiving hole on the gray circular component "
            "and insert it to complete the assembly."
        ),
    },
)
_CONFIGS.append(_tabero_v3_touch_20k)

# Paired whiteboard experiments. The observation rows and split are identical;
# only the absolute action-label source differs. Keep independent asset IDs so
# train-only action normalization can never leak across the comparison.
_TABERO_WHITEBOARD_TASK = (
    "Pick up the yellow whiteboard eraser and use it to erase the black X mark from the whiteboard."
)
_TABERO_WHITEBOARD_VALIDATION_EPISODES = (1, 7, 17)


def _make_tabero_whiteboard_config(
    *,
    name: str,
    dataset_name: str,
    action_source_mode: str,
    action_label_source: str,
    action_state_step_offset: int,
    tactile_lora_rank: int = 16,
    tactile_lora_alpha: float = 16.0,
    batch_size: int = 4,
    num_train_steps: int = 20_000,
    wait_for_checkpoint_on_save: bool = False,
) -> TrainConfig:
    return dataclasses.replace(
        _tabero_v3_touch_20k,
        name=name,
        model=dataclasses.replace(
            _tabero_v3_touch_20k.model,
            supervised_action_dim=None,
            tactile_loss_weight=TACTILE_LOSS_WEIGHT,
            padding_loss_weight=0.0,
            tactile_prefix_lora_rank=tactile_lora_rank,
            tactile_prefix_lora_alpha=tactile_lora_alpha,
        ),
        data=TaberoTacFieldWrenchDataConfig(
            repo_id=f"local/{dataset_name}",
            base_config=DataConfig(
                root=f"/data/yanghaojun/datasets/{dataset_name}",
                episodes=tuple(i for i in range(39) if i not in _TABERO_WHITEBOARD_VALIDATION_EPISODES),
                validation_episodes=_TABERO_WHITEBOARD_VALIDATION_EPISODES,
                video_backend="pyav",
                columns=(
                    "state",
                    "actions",
                    "wrist_wrench",
                    "tactile_marker_motion",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ),
                action_sequence_keys=("actions", "wrist_wrench"),
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
            use_so3_relative_actions=True,
        ),
        # Three seeded validation episodes contain 1,001 frames. Evaluate the
        # same 1,000 full-batch frames for both action-label definitions.
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        eval_num_batches=1_000 // batch_size,
        save_interval=4_000,
        keep_period=4_000,
        wait_for_checkpoint_on_save=wait_for_checkpoint_on_save,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-5,
            decay_steps=num_train_steps,
            decay_lr=1e-6,
        ),
        policy_metadata={
            **_tabero_local_smoke.policy_metadata,
            "dataset_version": dataset_name,
            "experiment_pair": "test2_whiteboard_action_source_ab",
            "action_source_mode": action_source_mode,
            "action_label_source": action_label_source,
            "action_state_step_offset": action_state_step_offset,
            "training_action_representation": "relative_xyz_so3_rotvec_absolute_gripper",
            "prediction_layout": "7d_action_then_6d_wrist_wrench",
            "output_action_dim": 7,
            "wrist_wrench_dim": 6,
            "predicts_wrench": True,
            "wrist_wrench_order": ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"],
            "wrist_wrench_target_alignment": "same_dataset_row_as_action_target",
            "wrist_wrench_source": "synchronized_robot_force_plus_robot_torque",
            "wrist_wrench_units": ["N", "N", "N", "N_m", "N_m", "N_m"],
            "wrist_wrench_frame": "K",
            "wrist_wrench_contract_source": "user_confirmed_2026-09-15",
            "wrist_wrench_loss_weight": TACTILE_LOSS_WEIGHT,
            "terminal_source_frame_omitted": True,
            "dataset_timing_policy": "compact",
            "dataset_missing_candidate_steps": 98,
            "validation_split_seed": 42,
            "validation_episode_ids": list(_TABERO_WHITEBOARD_VALIDATION_EPISODES),
            "tactile_adaptation": "tcn_lora",
            "tactile_lora_rank": tactile_lora_rank,
            "tactile_lora_alpha": tactile_lora_alpha,
            "checkpoint_mode": "wait_after_each_save" if wait_for_checkpoint_on_save else "async",
            "task_prompt": _TABERO_WHITEBOARD_TASK,
        },
    )


_TABERO_FULL_FP32_PARAMETER_POLICY = ParameterDtypePolicy(
    name="full_float32",
    default_trainable_dtype="float32",
)

_TABERO_MIXED_BF16_PARAMETER_POLICY = ParameterDtypePolicy(
    name="mixed_bfloat16",
    default_trainable_dtype="bfloat16",
    overrides=(
        # Keep normalization affine parameters in FP32 for numerical stability.
        ParameterDtypeRule(r"(?i).*norm.*/(?:bias|scale)", "float32"),
        # Vision patch/input embedding remains FP32; the rest of the vision transformer defaults to BF16.
        ParameterDtypeRule(r"PaliGemma/img/embedding/.*", "float32"),
        # State/action projections, time MLPs, and the action output head remain FP32.
        ParameterDtypeRule(
            r"(?:state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out|time_mlp_in|time_mlp_out)/.*",
            "float32",
        ),
        # First mixed-precision version deliberately keeps the sensor-specific tactile encoder in FP32.
        ParameterDtypeRule(r"tactile(?:_prefix)?_encoder/.*", "float32"),
    ),
)

_TABERO_MIXED_BF16_FP32_TRAIN_STATE_POLICY = dataclasses.replace(
    _TABERO_MIXED_BF16_PARAMETER_POLICY,
    name="mixed_bfloat16_fp32_train_state",
    gradient_dtype="float32",
    optimizer_state_dtype="float32",
)


def _make_tabero_whiteboard_full_finetune_config(
    *,
    name: str,
    dataset_name: str,
    action_source_mode: str,
    action_label_source: str,
    action_state_step_offset: int,
    parameter_dtype_policy: ParameterDtypePolicy = _TABERO_FULL_FP32_PARAMETER_POLICY,
    peak_lr: float = 2e-6,
    decay_lr: float = 2e-7,
) -> TrainConfig:
    """Build a strict Tabero-49999 full-parameter whiteboard config.

    The released checkpoint already contains backbone LoRA leaves, so its
    architecture is preserved. Unlike the adapter experiments, every leaf is
    trainable and no new tactile LoRA leaves are added: the pretrained tactile
    TCN itself is updated.
    """
    base = _make_tabero_whiteboard_config(
        name=name,
        dataset_name=dataset_name,
        action_source_mode=action_source_mode,
        action_label_source=action_label_source,
        action_state_step_offset=action_state_step_offset,
        tactile_lora_rank=0,
        batch_size=2,
        num_train_steps=12_000,
        wait_for_checkpoint_on_save=True,
    )
    return dataclasses.replace(
        base,
        # Strictly restore the exact released architecture, without randomly
        # initialized tactile adapters.
        weight_loader=_tabero_local_smoke.weight_loader,
        freeze_filter=nnx.Nothing,
        parameter_dtype_policy=parameter_dtype_policy,
        optimizer=_optimizer.AdamW(moment_dtype=parameter_dtype_policy.optimizer_state_dtype),
        ema_decay=None,
        fsdp_devices=2,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=peak_lr,
            decay_steps=12_000,
            decay_lr=decay_lr,
        ),
        policy_metadata={
            **base.policy_metadata,
            "optimization_method": "full_parameter_finetuning",
            "trainable_scope": "all_parameter_leaves",
            "pretrained_architecture": "tabero_49999_with_backbone_lora",
            "tactile_adaptation": "full_tcn_and_all_model_parameters",
            "tactile_lora_rank": 0,
            "tactile_lora_alpha": None,
            "parameter_dtype_policy": parameter_dtype_policy.name,
            "loss_reduction_dtype": "float32",
            "gradient_storage_dtype": parameter_dtype_policy.gradient_dtype,
            "gradient_cast_stage": (
                "after_autodiff_before_clipping_and_optimizer"
                if parameter_dtype_policy.gradient_dtype == "float32"
                else "none"
            ),
            "gradient_norm_and_clipping_dtype": "float32",
            "optimizer_state_dtype": parameter_dtype_policy.optimizer_state_dtype,
            "peak_lr": peak_lr,
            "decay_lr": decay_lr,
            "fsdp_devices": 2,
            "global_batch_size": 2,
        },
    )


def _make_tabero_whiteboard_full_finetune_sgd_config(
    *,
    name: str,
    dataset_name: str,
    action_source_mode: str,
    action_label_source: str,
    action_state_step_offset: int,
) -> TrainConfig:
    """Use no optimizer-sized tensors when two-device full AdamW does not fit."""
    base = _make_tabero_whiteboard_full_finetune_config(
        name=name,
        dataset_name=dataset_name,
        action_source_mode=action_source_mode,
        action_label_source=action_label_source,
        action_state_step_offset=action_state_step_offset,
    )
    return dataclasses.replace(
        base,
        optimizer=_optimizer.StatelessSGD(clip_gradient_norm=1.0),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-5,
            decay_steps=12_000,
            decay_lr=1e-6,
        ),
        policy_metadata={
            **base.policy_metadata,
            "optimizer": "stateless_sgd",
            "optimizer_state_strategy": "no_momentum_or_second_moment_tensors",
        },
    )


_CONFIGS.extend(
    [
        _make_tabero_whiteboard_config(
            name="pi0_lora_tabero_whiteboard_next_state_force_20k",
            dataset_name="test2_tabero_next_state_compact",
            action_source_mode="next-state",
            action_label_source="next_observation_state_within_episode",
            action_state_step_offset=1,
        ),
        _make_tabero_whiteboard_config(
            name="pi0_lora_tabero_whiteboard_sent_command_force_20k",
            dataset_name="test2_tabero_sent_command_compact",
            action_source_mode="sent-command",
            action_label_source="synchronized_absolute_sent_command_current_frame",
            action_state_step_offset=0,
        ),
        _make_tabero_whiteboard_config(
            name="pi0_lora_tabero_whiteboard_next_state_force_tactile_r32_12k",
            dataset_name="test2_tabero_next_state_compact",
            action_source_mode="next-state",
            action_label_source="next_observation_state_within_episode",
            action_state_step_offset=1,
            tactile_lora_rank=32,
            tactile_lora_alpha=32.0,
            batch_size=2,
            num_train_steps=12_000,
            wait_for_checkpoint_on_save=True,
        ),
        _make_tabero_whiteboard_config(
            name="pi0_lora_tabero_whiteboard_sent_command_force_tactile_r32_12k",
            dataset_name="test2_tabero_sent_command_compact",
            action_source_mode="sent-command",
            action_label_source="synchronized_absolute_sent_command_current_frame",
            action_state_step_offset=0,
            tactile_lora_rank=32,
            tactile_lora_alpha=32.0,
            batch_size=2,
            num_train_steps=12_000,
            wait_for_checkpoint_on_save=True,
        ),
        _make_tabero_whiteboard_full_finetune_config(
            name="pi0_tabero_whiteboard_next_state_force_full_ft_12k",
            dataset_name="test2_tabero_next_state_compact",
            action_source_mode="next-state",
            action_label_source="next_observation_state_within_episode",
            action_state_step_offset=1,
        ),
        _make_tabero_whiteboard_full_finetune_config(
            name="pi0_tabero_whiteboard_sent_command_force_full_ft_12k",
            dataset_name="test2_tabero_sent_command_compact",
            action_source_mode="sent-command",
            action_label_source="synchronized_absolute_sent_command_current_frame",
            action_state_step_offset=0,
        ),
        _make_tabero_whiteboard_full_finetune_config(
            name="pi0_tabero_whiteboard_next_state_force_full_ft_mixed_bf16_adamw_12k",
            dataset_name="test2_tabero_next_state_compact",
            action_source_mode="next-state",
            action_label_source="next_observation_state_within_episode",
            action_state_step_offset=1,
            parameter_dtype_policy=_TABERO_MIXED_BF16_FP32_TRAIN_STATE_POLICY,
            peak_lr=2e-5,
            decay_lr=2e-6,
        ),
        _make_tabero_whiteboard_full_finetune_config(
            name="pi0_tabero_whiteboard_sent_command_force_full_ft_mixed_bf16_adamw_12k",
            dataset_name="test2_tabero_sent_command_compact",
            action_source_mode="sent-command",
            action_label_source="synchronized_absolute_sent_command_current_frame",
            action_state_step_offset=0,
            parameter_dtype_policy=_TABERO_MIXED_BF16_FP32_TRAIN_STATE_POLICY,
            peak_lr=2e-5,
            decay_lr=2e-6,
        ),
        _make_tabero_whiteboard_full_finetune_config(
            name="pi0_tabero_whiteboard_next_state_force_full_ft_mixed_bf16_lowmem_adamw_12k",
            dataset_name="test2_tabero_next_state_compact",
            action_source_mode="next-state",
            action_label_source="next_observation_state_within_episode",
            action_state_step_offset=1,
            parameter_dtype_policy=_TABERO_MIXED_BF16_PARAMETER_POLICY,
            peak_lr=2e-5,
            decay_lr=2e-6,
        ),
        _make_tabero_whiteboard_full_finetune_config(
            name="pi0_tabero_whiteboard_sent_command_force_full_ft_mixed_bf16_lowmem_adamw_12k",
            dataset_name="test2_tabero_sent_command_compact",
            action_source_mode="sent-command",
            action_label_source="synchronized_absolute_sent_command_current_frame",
            action_state_step_offset=0,
            parameter_dtype_policy=_TABERO_MIXED_BF16_PARAMETER_POLICY,
            peak_lr=2e-5,
            decay_lr=2e-6,
        ),
        _make_tabero_whiteboard_full_finetune_sgd_config(
            name="pi0_tabero_whiteboard_next_state_force_full_ft_sgd_12k",
            dataset_name="test2_tabero_next_state_compact",
            action_source_mode="next-state",
            action_label_source="next_observation_state_within_episode",
            action_state_step_offset=1,
        ),
        _make_tabero_whiteboard_full_finetune_sgd_config(
            name="pi0_tabero_whiteboard_sent_command_force_full_ft_sgd_12k",
            dataset_name="test2_tabero_sent_command_compact",
            action_source_mode="sent-command",
            action_label_source="synchronized_absolute_sent_command_current_frame",
            action_state_step_offset=0,
        ),
    ]
)
_CONFIGS.append(
    dataclasses.replace(
        _tabero_touch_comparison,
        name="pi0_lora_tabero_rgb_state",
        model=dataclasses.replace(
            _tabero_touch_comparison.model,
            tactile_type=TactileType.NO,
            tactile_streams=(),
            tactile_dim_in=0,
            tactile_prefix_dim_in=0,
            tactile_prefix_history=None,
            tactile_prefix_lora_rank=0,
        ),
        data=TaberoNoTactNoForceDataConfig(
            repo_id=_tabero_local_smoke.data.repo_id,
            assets=_tabero_comparison_assets,
            base_config=dataclasses.replace(
                _tabero_local_smoke.data.base_config,
                columns=tuple(k for k in _tabero_local_smoke.data.base_config.columns if k != "tactile_marker_motion"),
            ),
            action_only=True,
            extra_delta_transform=True,
        ),
        weight_loader=dataclasses.replace(
            _tabero_local_smoke.weight_loader,
            strict_allow_extra_regex=(
                r"tactile_prefix_encoder/(blocks/block_[01]/kernels/kernel_[012]|"
                r"blocks/block_0/residual_proj|out_proj)/(kernel|bias)"
            ),
        ),
        policy_metadata={
            **_tabero_local_smoke.policy_metadata,
            "tactile_input": "none",
            "tactile_adaptation": "none",
            "tactile_lora_rank": 0,
        },
    )
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
