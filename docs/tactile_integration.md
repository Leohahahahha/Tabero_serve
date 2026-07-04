# How Tactile/Force Signals Are Introduced in This Project (Tabero / PI0 / PI05)

In this repository, tactile/force information is not just an extra input tensor. It is supported through two complementary paths:

- **As extra Transformer tokens**: the model supports an **encoder-prefix** tactile stream before the LLM and a **decoder-suffix** tactile stream before the action expert.
- **As part of the supervised action vector**: trailing action dimensions are treated as force/tactile torque slots, split from joint actions in the loss, and weighted separately.

The sections below use Tabero data as the running example and compare the changes with the official OpenPI PI0 (`pi0_base`) semantics.

---

## Data Side: Three Tactile Forms in Tabero

Tabero v2.1, as assumed by this repository, may contain:

- **`tactile_image`**: a third image stream mapped to `right_wrist_0_rgb`; this is effectively a visual modality.
- **`tactile_marker_motion`**: tactile force-field / marker-motion data, typically shaped **`[9, 198, 2]`**.
- **`tactile_gripper_force`**: gripper-force history, typically shaped **`[8, 6]`**.

In `src/openpi/policies/libero_policy.py`:

- `TaberoTacFieldInputs` reads `tactile_marker_motion`, reshapes it to **`[9, 198*2]`**, and writes it to `inputs["tactile_prefix"]`.
- `TaberoTacForceInputs` reads `tactile_gripper_force` and writes it to `inputs["tactile_suffix"]`.
- `TaberoTacAllInputs` uses three image streams and writes both `tactile_prefix` and `tactile_suffix`.

> **Normalization note:** the training pipeline runs `transforms.Normalize(norm_stats, ...)`. If statistics are computed with the updated `scripts/compute_norm_stats.py`, normalization is applied to `state/actions/tactile_prefix/tactile_suffix`. If tactile statistics are missing, for example when using older stats, tactile tensors remain in their original scale unless normalized, clipped, or unit-converted during data conversion.

---

## Model Side: Two Tactile Token Streams

The core model is implemented in `src/openpi/models/pi0.py`, and tactile encoders live in `src/openpi/models/tactile_encoder.py`.

### 0) Two Roles of Tactile/Force Signals

- **Conditioning information**: `Observation.tactile_prefix` / `Observation.tactile_suffix` is encoded into a single token and participates in self-attention, affecting action generation.
- **Supervision target**: trailing dimensions of `actions`, typically 6 dimensions, are treated as force/tactile torque slots and receive a separately weighted loss.

These two paths are independent. You can use only force-slot supervision without injecting tactile tokens, or inject tactile tokens without supervising force slots.

### 1) Prefix: Encoder-Prefix Tactile Token

This usually corresponds to **tacfield / marker motion**:

- Data enters `Observation.tactile_prefix`, with a shape close to `[B, 9, 396]`.
- The config sets `tactile_prefix_encoder_type="tcn"` in the Tabero configs in `src/openpi/training/config.py`.
- `TactileTCNEncoder` encodes the multi-frame sequence into **one embedding token** by taking the last hidden frame and applying a `Linear` projection.

### 2) Suffix: Decoder-Suffix Tactile Token

This usually corresponds to **tacforce / gripper-force history**:

- Data enters `Observation.tactile_suffix`, with a shape close to `[B, 8, 6]`.
- `MLPTactileEncoder` flattens all dimensions after the batch into an `in_dim` vector, for example `8*6=48`, and uses a two-layer MLP to produce **one embedding token**.
- The token is inserted into the suffix sequence, before state/action tokens in `Pi0.embed_suffix`.

---

## Computation: Tactile Encoding and Loss Splitting

### 1) Tactile Encoding

#### 1.1 MLP Tactile Encoder (Suffix Stream, Tacforce)

For an input `tactile_suffix` with shape `[B, 8, 6]`, the implementation flattens all non-batch dimensions into an `in_dim=48` vector, then applies two linear layers with a swish activation:

- \(x = \mathrm{flatten}(\mathrm{tactile\_suffix}) \in \mathbb{R}^{B\times 48}\)
- \(h = \mathrm{swish}(W_1 x + b_1)\)
- \(z = W_2 h + b_2\), producing \(z \in \mathbb{R}^{B\times d_{\text{emb}}}\)

In practice, this is a lightweight temporal aggregator that compresses a short tactile-history window into one conditioning token placed at the front of the action expert suffix sequence.

#### 1.2 TCN Tactile Encoder (Prefix Stream, Tacfield)

For an input `tactile_prefix` with shape `[B, 9, 396]`, derived from Tabero marker motion `[9, 198, 2]`, the TCN block implements causal convolution explicitly. For each timestep \(t\), it aggregates the previous \(k\) offsets with linear kernels and adds a residual:

- Let \(\tilde{x}\) be the causally padded sequence.
- \(y_t=\sum_{i=0}^{k-1} W_i \tilde{x}_{t-i}\)
- \(r_t=x_t\) or \(r_t=W_r x_t\) when dimensions differ.
- Output: \(\mathrm{swish}(y_t+r_t)\)

After several layers, the TCN encoder takes the final hidden timestep \(h_T\) and projects it into a single token:

- \(z=W_o h_T+b_o \in \mathbb{R}^{B\times d_{\text{emb}}}\)

This emphasizes temporal dynamics in marker motion and is therefore better suited for the prefix stream, where it is aligned with image and language tokens.

### 2) Joint Action/Force Prediction and Loss

When `Pi0Config.tactile_type = EXPERT_HIS_C_FUT`, `actions` already contains `[joint action, force/tactile torque]`, for example 13 dimensions: 7 joint dimensions plus 6 force dimensions.

- `effective_action_dim` tells the model how many dimensions are semantically valid before padding to a larger `action_dim`.
- `tactile_dim` specifies the number of force/tactile slots, commonly 6.
- `action_loss` is computed only on the first `ctrl_dim = effective_action_dim - tactile_dim` dimensions.
- `tactile_loss` is computed on the following `tactile_dim` dimensions.
- `total_loss = action_loss + tactile_loss_weight * tactile_loss`.

This explains two common baselines:

- **no-force**: `tactile_loss_weight=0.0`; the model may still output 13D actions, but force dimensions do not contribute to the loss.
- **only-joint**: data-side `SliceActions(7)` drops the force dimensions entirely.

More formally, PI0/PI05 use a flow-matching / diffusion-style regression objective. This project's action/force split applies different weights to different subspaces of the same prediction vector: joint dimensions weight 1, force-slot dimensions weight `tactile_loss_weight`, and padding dimensions weight 0 unless configured otherwise.

---

## Comparison with Official OpenPI PI0 (`pi0_base`)

Here, "official OpenPI PI0" refers to the released `pi0_base` training/inference semantics: image + language + low-dimensional state conditions produce future action sequences, without tactile tokens and without force/tactile slots in the supervised output. In inference, only the first 7 joint-action dimensions are commonly returned.

| Aspect | Official OpenPI PI0 (`pi0_base` semantics) | This project |
| --- | --- | --- |
| Image input | Usually 1-2 streams; missing views use padding + masks | Explicitly supports three streams: `image/wrist_image/tactile_image` mapped to base/left/right |
| Tactile input | No tactile fields | Adds optional `Observation.tactile_prefix` / `Observation.tactile_suffix` |
| State input | PI0 uses continuous state via `state_proj` | PI0 is the same; PI05 does not add an explicit continuous state token |
| Action output semantics | Usually only the first 7 joint-action dimensions are used | Supports **13D (7+6)** outputs, where the last 6 dimensions are supervisable and returnable force/tactile slots |

Inference output trimming happens in `src/openpi/policies/libero_policy.py`:

- `LiberoOutputs` returns the first 7 joint-action dimensions.
- `LiberoForceOutputs` returns the first 13 dimensions, including force slots.

### Token Sequence Structure

| Aspect | Official OpenPI PI0 (`pi0_base` semantics) | This project |
| --- | --- | --- |
| Prefix sequence | `ImgTokens + TextTokens` | `ImgTokens + TextTokens + optional tactile_prefix token` |
| Suffix sequence (PI0) | `state_token + ActionTokens` | `optional tactile_suffix token + state_token + ActionTokens` |
| Suffix sequence (PI05) | `ActionTokens`, without explicit state token | `optional tactile_suffix token + ActionTokens` |
| Tactile influence | None | Tactile tokens affect later action generation through self-attention |

The insertion points are in `src/openpi/models/pi0.py`: prefix tactile tokens are appended after image/language tokens, and suffix tactile tokens are inserted before state/action tokens.

### Training Objective Difference

Official PI0 mainly treats the action vector as joint-action semantics. This project splits the action space into \([a, f]\), joint actions and force slots, and assigns separate weights. `effective_action_dim` also removes padding dimensions from the loss unless `padding_loss_weight` is enabled.

### Padding Dimensions

When `action_dim` is fixed, for example 32, but real data has only 7, 13, or 14 valid dimensions, `PadStatesAndActions` fills the remaining dimensions with zeros. This project supports two strategies:

- **Masked padding loss**: ignore padding dimensions entirely.
- **Unmasked padding loss**: include padding dimensions in the loss and push them toward zero, which can better match the original checkpoint training style.

Set `Pi0Config.padding_loss_weight` to a nonzero value to enable padding loss.

---

## Special Case: Force-Slot Supervision Without Tactile Tokens

This project can enable the split supervision logic without creating tactile token encoder weights:

- `tactile_type=EXPERT_HIS_C_FUT`: enable action/force split loss.
- `tactile_dim_in=0`: do not create a suffix tactile encoder.
- `tactile_streams=()`: do not inject prefix or suffix tactile tokens.

This pattern is used by several Tabero configs such as `pi0_lora_tacimg_tabero`, `pi0_lora_notac_tabero`, and `pi05_notac_tabero`.

---

## Summary Compared with Traditional PI0

Traditional PI0, in this repository's terminology, uses images, continuous state, and a language prompt to produce a padded action vector. It has no `tactile_prefix` or `tactile_suffix` tokens and no action-vs-force split weighting.

The tactile PI0/PI05 variants add two optional tactile token streams:

- prefix: tacfield, marker motion encoded by TCN into one token;
- suffix: tacforce, 8x6 gripper force encoded by MLP into one token.

They also explicitly include force/tactile slots inside the action vector, for example 13D, and use `effective_action_dim`, `tactile_dim`, and `tactile_loss_weight` to split and weight the loss.

PI05 differs from PI0 in state handling and timestep injection: PI05 routes state through discrete/language-side configuration and uses adaRMS for timestep conditioning, while PI0 has an explicit continuous `state token` and mixes time/action with an MLP.

---

## Mermaid Diagram: How Tactile Signals Enter the Model

```mermaid
flowchart LR
  subgraph Dataset[Tabero sample flat fields]
    IMG1[image]
    IMG2[wrist_image]
    IMG3[tactile_image]
    MM[tactile_marker_motion<br/>[9,198,2]]
    GF[tactile_gripper_force<br/>[8,6]]
    ST[state]
    AC[actions<br/>[13]=7 joint + 6 force]
  end

  subgraph Inputs[policy input mapping in libero_policy]
    IIMG[images: base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb]
    ITAC_P[tactile_prefix<br/>reshape: [9,198*2]=[9,396]]
    ITAC_S[tactile_suffix<br/>[8,6]]
    IST[state]
    IAC[actions]
  end

  subgraph Enc[tactile encoders in tactile_encoder]
    TCN[TCN encoder<br/>[B,9,396] -> 1 token]
    MLP[MLP encoder<br/>flatten 48 -> 1 token]
  end

  subgraph Model[Pi0 / Pi05 in pi0.py]
    PFX[Prefix tokens<br/>image + text (+ tactile_prefix token)]
    SFX[Suffix tokens<br/>(+ tactile_suffix token) (+ state token for Pi0) + action tokens]
    LLM[Gemma Transformer]
    OUT[action_out_proj -> v_t]
    LOSS[Loss split (EXPERT_HIS_C_FUT)<br/>action_loss + w * tactile_loss]
  end

  IMG1 --> IIMG
  IMG2 --> IIMG
  IMG3 --> IIMG
  MM --> ITAC_P --> TCN --> PFX
  GF --> ITAC_S --> MLP --> SFX
  ST --> IST --> SFX
  AC --> IAC --> LOSS

  IIMG --> PFX --> LLM --> OUT --> LOSS
  SFX --> LLM
```
