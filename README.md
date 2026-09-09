# Tabero-VTLA

Tabero-VTLA is a vision-tactile-language-action (VTLA) model training and inference repository deeply customized from the open-source [openpi framework](https://www.physicalintelligence.company/) by the Physical Intelligence team. This repository focuses on **robot manipulation policy learning with tactile/force information fusion** and fine-tuning on the Tabero tactile dataset.

This repository currently supports the following models:
- [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-matching-based vision-language-action model (VLA)
- [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with stronger open-world generalization

The core feature of this repository is **tactile/force modality fusion**. On top of the original VLA model, it adds support for tactile modalities such as tactile force fields (marker motion) and gripper force history, and it supports explicit supervision of force/torque slots in the action space.

## Requirements

FR3 deployment with ZED, D405 and DM-Tac W: see the [Chinese deployment guide](docs/tabero_fr3_deployment.md)
for the model server, ROS2 client, observation contracts and manual shadow/execution commands.

Running the models in this repository requires an NVIDIA GPU. The estimated requirements are listed below (single-GPU estimates; multi-GPU model parallelism can also be configured through `fsdp_devices` to reduce per-GPU memory usage):

| Mode | VRAM Requirement | Recommended GPU |
| ---- | ---------------- | --------------- |
| Inference | > 8 GB | RTX 4090 |
| Fine-tuning (LoRA) | > 22.5 GB | RTX 4090 |
| Fine-tuning (full parameters) | > 70 GB | A100 (80GB) / H100 |

This repository has been tested on Ubuntu 22.04.

## Installation

When cloning this repository, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:NathanWu7/Tabero-VTLA.git

# If the repository has already been cloned:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. Please follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to install it. After installation, run the following commands to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Note: `GIT_LFS_SKIP_SMUDGE=1` is needed when pulling LeRobot as a dependency.

## Model Checkpoints

### Base Models

We provide several base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data and can be used for fine-tuning:

| Model | Use | Description | Checkpoint Path |
| ----- | --- | ----------- | --------------- |
| π₀ | Fine-tuning | Base π₀ model | `gs://openpi-assets/checkpoints/pi0_base` |
| π₀.₅ | Fine-tuning | Base π₀.₅ model | `gs://openpi-assets/checkpoints/pi05_base` |

Checkpoints are downloaded automatically from `gs://openpi-assets` by default and cached under `~/.cache/openpi`. You can override the download path by setting the `OPENPI_DATA_HOME` environment variable.

## Tactile/Force Fusion Overview

This repository introduces two tactile/force information fusion paths on top of the original VLA model:

- **As extra Transformer tokens**: supports two tactile streams, **encoder-prefix** (before entering the LLM) and **decoder-suffix** (before entering the action expert).
- **As part of the supervised action vector**: treats the trailing dimensions of the action vector as "force/tactile torque slots" and splits them from joint actions in the loss with separate weights.

For details, see the [tactile integration documentation](docs/tactile_integration.md).

## Quick Start: Fine-Tune π₀ on Tabero Data

For the local FR3 **real-robot, marker-motion input / action-only LoRA continuation**
configuration, see [the code-only preparation guide](docs/tabero_real_robot_smoke.md).
It uses an existing Tabero checkpoint, a 26/3 episode split and only 7D action supervision.
The guide includes a frozen-tactile baseline and an opt-in **backbone + tactile TCN LoRA**
variant for DM-Tac sensor adaptation; training and model evaluation remain user-operated.
For a separately trained RGB+state-only baseline and paired offline comparison, see
[the no-tactile retraining guide](docs/tabero_rgb_state_baseline.md).
For the v3 next-state labels, fresh published-Tabero initialization, 20k tactile-LoRA
schedule, and W&B handoff, see [the v3 tactile 20k guide](docs/tabero_v3_touch_20k.md).
For a dated engineering retrospective covering data, training, evaluation, and
real-robot deployment incidents, see [the interview incident log](docs/tabero_interview_incident_log.md).
The original example below starts from the base model and is not that configuration.

Using `pi0_lora_tacfield_tabero` as an example (two image streams + tactile force field + 13D joint action/force prediction), training has three steps:

### 1. Prepare Data

The training configurations in this repository directly use the LeRobot-format Tabero dataset on Hugging Face (`NathanWu7/tabero`), so no manual data conversion is required.

### 2. Compute Normalization Statistics and Start Training

First compute normalization statistics:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacfield_tabero
```

Then start training:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_lora_tacfield_tabero --exp-name=my_experiment --overwrite
```

Training progress is printed to the console, and checkpoints are saved under the `checkpoints` directory. You can also monitor training through the Weights & Biases dashboard. To maximize GPU memory utilization, we recommend setting `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.

> **Note**: All available training configurations can be found in [src/openpi/training/config.py](src/openpi/training/config.py), including different tactile modality combinations such as tacimg, tacfield, tacforce, and tacall.

### 3. Start the Policy Server for Inference

After training, you can start the policy server for inference. The following example uses the checkpoint at iteration 50000:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_lora_tacfield_tabero --policy.dir=checkpoints/pi0_lora_tacfield_tabero/my_experiment/50000
```

This starts a WebSocket server on port 8000 and waits for incoming observations for inference.

You can also use the convenience script:

```bash
bash server.sh pi0_lora_tacfield_tabero 50000
```

## LIBERO Example

We provide an example for fine-tuning on the LIBERO simulation benchmark:

```bash
# Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_libero_exp --overwrite
```

## PyTorch Support

This repository also provides PyTorch implementations of the π₀ and π₀.₅ models. The PyTorch implementation has been validated on the LIBERO benchmark for both inference and fine-tuning.

### Environment Setup

1. Make sure the latest dependencies are installed: `uv sync`
2. Confirm that the transformers version is 4.53.2: `uv pip show transformers`
3. Apply the transformers library patch:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

### Convert a JAX Model to PyTorch

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name pi0_lora_tacfield_tabero \
    --output_path /path/to/converted/pytorch/checkpoint
```

### PyTorch Training

```bash
# Single-GPU training:
uv run scripts/train_pytorch.py pi0_lora_tacfield_tabero --exp_name pytorch_test

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_lora_tacfield_tabero --exp_name pytorch_ddp_test
```

## FAQ

| Question | Solution |
| -------- | -------- |
| `uv sync` fails due to dependency conflicts | Try deleting the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If the issue persists, check whether the latest version of `uv` is installed (`uv self update`). |
| GPU memory is insufficient during training | Make sure `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) is set so JAX can use more GPU memory. You can also enable fully sharded data parallelism with `--fsdp-devices <n>`. If memory is still insufficient, consider disabling EMA. |
| Policy server connection error | Check whether the server is running on the expected port. Confirm the network connection and firewall settings between the client and server. |
| Training fails because norm stats cannot be found | Run `scripts/compute_norm_stats.py` before training to compute normalization statistics. |
| Dataset download fails | Check the network connection. For Hugging Face datasets, make sure you are logged in (`huggingface-cli login`). |
| CUDA/GPU error | Verify that the NVIDIA driver is installed correctly. System-level CUDA libraries are not required because they are installed through uv. If CUDA issues occur, you can even try uninstalling system CUDA libraries because they sometimes cause conflicts. |
| Action dimension mismatch | Verify that the data processing transforms match the expected input/output dimensions of your robot. Check the action space definition in the policy classes. |
| Training loss diverges | Check the `q01`, `q99`, and `std` values in the dataset's `norm_stats.json`. Some rarely used dimensions may have very small `q01`, `q99`, or `std` values, causing normalized state and action values to become huge. You can manually adjust norm stats as a temporary workaround. |

## License

This project is open-sourced under the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This repository is customized from the [openpi](https://github.com/Physical-Intelligence/openpi) project by [Physical Intelligence](https://www.physicalintelligence.company/). We thank the original authors for their excellent work.
