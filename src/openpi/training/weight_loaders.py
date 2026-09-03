import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Regex specifying parameters that are missing from the checkpoint but exist in the current model
    # and should be filled from the current model parameters. By default, only LoRA weights are filled, preserving previous behavior.
    missing_regex: str = ".*lora.*"
    strict: bool = False
    # Opt-in exception for genuinely new leaves only; existing/extra shapes remain strict.
    # Kept separate from missing_regex so strict=True never implicitly permits missing LoRA.
    strict_allow_missing_regex: str | None = None
    # Explicit removed-module exception; shared and unexpected leaves stay strict.
    strict_allow_extra_regex: str | None = None

    def __post_init__(self):
        if self.strict_allow_missing_regex is not None and not self.strict:
            raise ValueError("strict_allow_missing_regex requires strict=True")
        if self.strict_allow_extra_regex is not None and not self.strict:
            raise ValueError("strict_allow_extra_regex requires strict=True")

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        if self.strict:
            expected = flax.traverse_util.flatten_dict(params, sep="/")
            actual = flax.traverse_util.flatten_dict(loaded_params, sep="/")
            missing, extra = set(expected) - set(actual), set(actual) - set(expected)
            allowed_extra = {
                k for k in extra
                if self.strict_allow_extra_regex is not None and re.fullmatch(self.strict_allow_extra_regex, k)
            }
            extra -= allowed_extra
            allowed_missing = {
                k for k in missing
                if self.strict_allow_missing_regex is not None and re.fullmatch(self.strict_allow_missing_regex, k)
            }
            if allowed_missing and any(
                re.fullmatch(self.strict_allow_missing_regex, k) for k in actual
            ):
                raise ValueError("Strict checkpoint mismatch: partially present new adapters; refusing to reset them")
            missing -= allowed_missing
            mismatched = {k for k in expected.keys() & actual.keys() if expected[k].shape != actual[k].shape}
            if missing or extra or mismatched:
                raise ValueError(
                    f"Strict checkpoint mismatch: missing={sorted(missing)}, extra={sorted(extra)}, "
                    f"shape_mismatch={sorted(mismatched)}"
                )
            logger.info("Strictly restored %d leaves (%d LoRA) from %s", len(actual) - len(allowed_extra),
                        sum("lora" in k for k in actual if k not in allowed_extra), self.params_path)
            if allowed_extra:
                logger.info("Discard %d explicitly removed-module leaves: %s", len(allowed_extra), sorted(allowed_extra))
            if allowed_missing:
                logger.info("Initialize %d explicitly allowed new adapter leaves: %s",
                            len(allowed_missing), sorted(allowed_missing))
        # Add all missing weights that match missing_regex, such as LoRA weights or weights from newly added modules.
        missing_regex = (self.strict_allow_missing_regex or "(?!)") if self.strict else self.missing_regex
        return _merge_params(loaded_params, params, missing_regex=missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def merge_loaded_params(loaded_params: at.Params, ref_params: at.Params, *, missing_regex: str = ".*") -> at.Params:
    """Public wrapper for merging checkpoint params into a reference param tree.

    This is useful outside of training (e.g., serving-time weight composition).

    - All keys that exist in both `loaded_params` and `ref_params` are taken from `loaded_params`.
    - Any keys missing from `loaded_params` that match `missing_regex` are filled from `ref_params`.
    """
    return _merge_params(loaded_params, ref_params, missing_regex=missing_regex)


def override_params_by_regex(
    base_params: at.Params, overlay_params: at.Params, *, allow_regex: str
) -> at.Params:
    """Override a subset of params (selected by regex) from overlay into base.

    Only keys that:
    - match `allow_regex`, AND
    - exist in both base and overlay
    will be overridden.
    """
    pattern = re.compile(allow_regex)
    flat_base = flax.traverse_util.flatten_dict(base_params, sep="/")
    flat_overlay = flax.traverse_util.flatten_dict(overlay_params, sep="/")

    for k, base_v in list(flat_base.items()):
        if not pattern.fullmatch(k):
            continue
        if k not in flat_overlay:
            continue
        ov = flat_overlay[k]
        flat_base[k] = ov.astype(base_v.dtype) if getattr(ov, "dtype", None) is not None and ov.dtype != base_v.dtype else ov

    return flax.traverse_util.unflatten_dict(flat_base, sep="/")
