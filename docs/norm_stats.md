# Normalization Statistics

During policy training and inference, our models normalize proprioceptive state inputs and action targets. The statistics used for normalization are computed on the training data and stored together with model checkpoints.

## Reloading Normalization Statistics

When fine-tuning our models on a new dataset, you need to decide whether to (A) reuse existing normalization statistics or (B) compute new statistics on the new training data. The better choice depends on how similar your robot and task are to the robot and task distribution in the pre-training data.

**If your target robot matches the pre-trained statistics, we recommend reloading the same normalization statistics.** By reloading normalization statistics, the actions in the dataset will be more "familiar" to the model, which may improve performance.

**Note:** Whether reloading normalization statistics is beneficial depends on how similar your robot and task are to the robot and task distribution in the pre-training data. We recommend always trying both options, reloading existing statistics and computing new statistics on the new dataset (see [README](../README.md) for how to compute new statistics), then choosing the one that works better for your task.

## Provided Pre-Trained Normalization Statistics

Below is a list of all pre-trained normalization statistics we provide. These statistics are available for both the `pi0_base` and `pi0_fast_base` models. For `pi0_base`, set `assets_dir` to `gs://openpi-assets/checkpoints/pi0_base/assets`; for `pi0_fast_base`, set `assets_dir` to `gs://openpi-assets/checkpoints/pi0_fast_base/assets`.

| Robot | Description | Asset ID |
| ----- | ----------- | -------- |
| ALOHA | 6-DoF dual-arm robot with parallel grippers | trossen |
| Mobile ALOHA | Mobile version of ALOHA mounted on a Slate base | trossen_mobile |
| ARX | Dual-arm ARX-5 robot with parallel grippers | arx |
| ARX mobile | Mobile version of dual-arm ARX-5 mounted on a Slate base | arx_mobile |
| Fibocom mobile | Fibocom mobile robot with two ARX-5 arms | fibocom_mobile |

## Pi0 Model Action Space Definition

`pi0_base` and `pi0_fast_base` use the following action space definition. Left and right are defined from the rear of the robot facing the workspace:

```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left gripper position",
    "dim_7:dim_12": "right arm joint angles (dual-arm only)",
    "dim_13": "right gripper position (dual-arm only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (mobile robots only)",
```

The proprioceptive state uses the same definition as the action space, except that the mobile robot's base x-y position (the last two dimensions) is not included in the proprioceptive state.

For 7-DoF robots, we use the first 7 dimensions of the action space as joint actions and the 8th dimension as the gripper action.

General information:
- Joint angles are represented in radians. The zero position corresponds to the zero position reported by each robot interface library.
- Gripper positions are in the range [0.0, 1.0], where 0.0 means fully open and 1.0 means fully closed.
- The control frequency is 20 Hz or 50 Hz, depending on the robot platform.
