import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        #
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Optional: third camera (e.g., tactile image) mapped to the "right wrist" slot.
        # If not provided, we fall back to zeros like the original LiberoInputs.
        if "observation/tactile_image" in data:
            right_image = _parse_image(data["observation/tactile_image"])
            right_mask = np.True_
        else:
            right_image = np.zeros_like(base_image)
            # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
            right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Either real third view (e.g. tactile image) or zero padding.
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # Optional: multi-frame gripper / tactile force as tactile / torque input.
        # Tactile/force stream configuration and loss behavior.
        # Implementation note.
        if "gripper_force" in data:
            inputs["tactile_suffix"] = data["gripper_force"]

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacImgInputs(transforms.DataTransformFn):
    """
    Tactile/force stream configuration and loss behavior.

    Image stream mapping and masking behavior.
        - observation/image          -> base_0_rgb
        - observation/wrist_image    -> left_wrist_0_rgb
        - observation/tactile_image  -> right_wrist_0_rgb
    Action/force dimensions and loss handling.
        Action/force dimensions and loss handling.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Dataset format and transform behavior.
        #   "image", "wrist_image", "tactile_image", "state", "actions", ...
        # Implementation note.
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        tactile_image = _parse_image(data["tactile_image"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": tactile_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Tactile/force stream configuration and loss behavior.
        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacFieldInputs(transforms.DataTransformFn):
    """
    Tactile/force stream configuration and loss behavior.

    Image stream mapping and masking behavior.
        - image                  -> base_0_rgb
        - wrist_image            -> left_wrist_0_rgb
      Implementation note.
    Tactile/force stream configuration and loss behavior.
        Tactile/force stream configuration and loss behavior.
          Tactile/force stream configuration and loss behavior.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Implementation note.
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        state = data["state"]

        right_image = np.zeros_like(base_image)
        right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        if "tactile_marker_motion" not in data:
            raise KeyError("TaberoTacFieldInputs expects 'tactile_marker_motion' in data.")
        motion = np.asarray(data["tactile_marker_motion"])
        if motion.ndim != 3:
            raise ValueError(f"tactile_marker_motion must be 3D, got shape {motion.shape}")
        n, m, d = motion.shape
        tactile = motion.reshape(n, m * d)
        inputs["tactile_prefix"] = tactile

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacForceInputs(transforms.DataTransformFn):
    """
    Tactile/force stream configuration and loss behavior.

    Image stream mapping and masking behavior.
        - image                  -> base_0_rgb
        - wrist_image            -> left_wrist_0_rgb
      Implementation note.
    Tactile/force stream configuration and loss behavior.
        Tactile/force stream configuration and loss behavior.
          Tactile/force stream configuration and loss behavior.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Implementation note.
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        state = data["state"]

        right_image = np.zeros_like(base_image)
        right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        if "tactile_gripper_force" not in data:
            raise KeyError("TaberoTacForceInputs expects 'tactile_gripper_force' in data.")
        inputs["tactile_suffix"] = data["tactile_gripper_force"]

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacForceEncInputs(transforms.DataTransformFn):
    """
    Tactile/force stream configuration and loss behavior.

    Implementation note.
    Image stream mapping and masking behavior.
    Tactile/force stream configuration and loss behavior.
      Tactile/force stream configuration and loss behavior.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Implementation note.
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        state = data["state"]

        right_image = np.zeros_like(base_image)
        right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # Tactile/force stream configuration and loss behavior.
        # Tactile/force stream configuration and loss behavior.
        if "tactile_gripper_force" not in data:
            raise KeyError("TaberoTacForceEncInputs expects 'tactile_gripper_force' in data.")
        inputs["tactile_prefix"] = data["tactile_gripper_force"]

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacAllInputs(transforms.DataTransformFn):
    """
    Tactile/force stream configuration and loss behavior.

    Tactile/force stream configuration and loss behavior.
    Encoder configuration and sequence handling.
        Tactile/force stream configuration and loss behavior.
    Implementation note.
        Tactile/force stream configuration and loss behavior.
    Action/force dimensions and loss handling.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Implementation note.
        #   image / wrist_image / tactile_image / state / tactile_marker_motion / tactile_gripper_force / actions / prompt
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        tactile_image = _parse_image(data["tactile_image"])
        state = data["state"]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": tactile_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Tactile/force stream configuration and loss behavior.
        motion = np.asarray(data["tactile_marker_motion"])
        if motion.ndim != 3:
            raise ValueError(f"tactile_marker_motion must be 3D, got shape {motion.shape}")
        n, m, d = motion.shape
        tactile_prefix = motion.reshape(n, m * d)
        inputs["tactile_prefix"] = tactile_prefix

        # Tactile/force stream configuration and loss behavior.
        inputs["tactile_suffix"] = data["tactile_gripper_force"]

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoNoTactInputs(transforms.DataTransformFn):
    """
    Tactile/force stream configuration and loss behavior.

    Image stream mapping and masking behavior.
        - image                  -> base_0_rgb
        - wrist_image            -> left_wrist_0_rgb
      Implementation note.
    Tactile/force stream configuration and loss behavior.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Implementation note.
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        state = data["state"]

        right_image = np.zeros_like(base_image)
        right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # Action/force dimensions and loss handling.
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LiberoForceOutputs(transforms.DataTransformFn):
    """
    Tactile/force stream configuration and loss behavior.
    """

    def __call__(self, data: dict) -> dict:
        # Action/force dimensions and loss handling.
        return {"actions": np.asarray(data["actions"][:, :13])}
