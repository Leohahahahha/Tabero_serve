# Tabero v3 tactile LoRA 20k training

This profile is `pi0_lora_tabero_v3_touch_20k`. It uses RGB, wrist RGB, state, and the `[9,198,2]` marker-coordinate history; it predicts only the 7D action. It restores the published Tabero checkpoint at step 49999 and creates a new optimizer. It does not restore any local smoke, recovery, or 2999 checkpoint.

The v3-only action transform is translation relative to the current state, true SO(3) relative rotation `R_relative = inverse(R_state) * R_action`, and an absolute single-finger gripper coordinate. Inference applies the exact inverse `R_action = R_state * R_relative` before returning the same absolute 7D command expected by the real-robot code. Older training configurations keep their legacy component subtraction.

The fixed split is 26 training episodes (5,936 frames) and validation episodes `4,14,24` (690 frames). With global batch 4, each validation pass covers 688 frames. Training is 20,000 updates, approximately 13.5 passes over the training anchors. Validation runs every 1,000 updates. Checkpoints are retained at `4000,8000,12000,16000,20000`.

The v3 label contract was verified read-only: for every non-terminal row, `actions[t] == state[t+1]`. Although 114 raw axis-angle component jumps remain in the stored data, their largest physical one-step rotation is only 1.2253 degrees; the SO(3) training transform therefore removes the false approximately `2*pi` residuals. The loader obtains prompt text through each Parquet row's `task_index` and the authoritative `meta/tasks.jsonl` mapping. That mapping contains:

```text
Align the black circular component with the receiving hole on the gray circular component and insert it to complete the assembly.
```

The per-episode summary strings in `episodes.jsonl` still contain the older acquisition description. The preparation report records all such mismatches. They do not change the prompt consumed by this loader, but the converter should synchronize them in a future dataset revision.

Two limitations are recorded, not silently changed. Twelve episodes contain 18 compacted missing sample intervals; the missing sensor observations and their exact output transition positions cannot be reconstructed from this compact dataset, so a later strict export should split at gaps. Also, 2,726/6,626 (41.14%) one-step expert translations exceed 2 mm, whereas the initial real-robot guard permits only 2 mm per 100 ms. Training labels are intentionally not clipped to that provisional safety guard: clipping a 50-step expert trajectory would change the learned task and does not recover closed-loop lag. Keep the guard for initial shadow/guarded deployment, measure tracking lag, and only tune it under a separately reviewed robot-safety procedure.

## One-time preparation

Run every command below as one physical shell line. Do not paste Bash `\` continuations on this workstation because its injected command-recording library can abort the terminal.

```bash
cd /home/yanghaojun/Tabero-VTLA
```

```bash
source /data/yanghaojun/envs/tabero-smoke/bin/activate
```

```bash
export HF_HOME=/data/yanghaojun/cache/huggingface
```

```bash
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
```

Generate v3 train-only normalization statistics and decode-audit every video. This is CPU data preparation, not model training:

```bash
JAX_PLATFORMS=cpu python scripts/prepare_tabero_smoke.py --config=pi0_lora_tabero_v3_touch_20k --output-dir=/data/yanghaojun/outputs/pi0_lora_tabero_v3_touch_20k-audit
```

The required files should then exist:

```bash
test -f /data/yanghaojun/outputs/assets/pi0_lora_tabero_v3_touch_20k/local/tabero_lerobot_compact_v3/norm_stats.json && test -f /data/yanghaojun/outputs/assets/pi0_lora_tabero_v3_touch_20k/local/tabero_lerobot_compact_v3/split_provenance.json && echo READY
```

Log in without putting the API key directly into shell history:

```bash
wandb login --verify
```

Optionally choose a W&B team/entity and a different project name:

```bash
export WANDB_ENTITY=your_wandb_team_or_username
```

```bash
export TABERO_WANDB_PROJECT=tabero-vtla
```

By default W&B receives the run configuration, scalar training metrics every 10 updates, and validation metrics every 1,000 updates. Camera samples are not uploaded. To explicitly allow a first-batch camera montage, run `export TABERO_WANDB_LOG_IMAGES=1` before launch.

## Launch

First inspect current GPU availability:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

Choose one, two, or four free cards. Four cards use data parallelism with one sample per GPU while preserving global batch 4:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

Use a unique run name containing only letters, digits, `_`, and `-`:

```bash
export TABERO_RUN=v3_touch_20k_run1
```

The dry run validates paths and prints the exact command without initializing JAX, W&B, or training:

```bash
bash scripts/run_tabero_v3_touch_20k.sh --dry-run "$TABERO_RUN"
```

Start the actual run:

```bash
bash scripts/run_tabero_v3_touch_20k.sh "$TABERO_RUN"
```

The launcher sets W&B to online mode and writes a local copy of all console output. Do not add `--no-wandb-enabled`.

## Monitor and resume

In another terminal, follow the console log:

```bash
tail -n 100 -f "/data/yanghaojun/outputs/logs/${TABERO_RUN}.log"
```

Inspect structured train/validation records:

```bash
tail -n 20 "/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_v3_touch_20k/${TABERO_RUN}/metrics.jsonl"
```

Inspect GPU utilization:

```bash
watch -n 2 nvidia-smi
```

The console prints the W&B run URL. The same run ID is stored at:

```text
/data/yanghaojun/outputs/checkpoints/pi0_lora_tabero_v3_touch_20k/<RUN_NAME>/wandb_id.txt
```

If the process stops after at least one completed checkpoint, keep the same GPU selection and run name, then resume the optimizer, model, data-loader state, and same W&B run:

```bash
bash scripts/run_tabero_v3_touch_20k.sh --resume "$TABERO_RUN"
```

Do not use `--resume` to change the split, batch, learning-rate schedule, or initialization. A completed run ends with checkpoint `20000`.

This is still offline imitation training. A lower validation loss does not establish safe or successful real-robot assembly; run matched offline evaluation and guarded shadow deployment before enabling commands.
