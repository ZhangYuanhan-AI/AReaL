# Fine-tuning Large MoE Models

Compared to PyTorch FSDP, Megatron-LM supports full 5D parallelism, delivering better
scaling and efficiency. AReaL fully supports customized RL training with Megatron-LM as
the backend. This guide explains how to harness the Megatron training backend and train
large MoE models for your application.

## Enabling Megatron Backend

Shifting from FSDP to Megatron requires only a single line of change: the
`actor.backend` field from `fsdp:d4` to `megatron:d4`.

For a complete guide on allocation mode syntax, parallelism dimensions, and GPU
calculations, see the [Allocation Mode Reference](../reference/alloc_mode.md).

## MoE Parallel Strategy

For MoE models, Megatron supports separate parallelism for attention and FFN modules
using the hybrid syntax. For example:

```
megatron:(attn:d1p4t2c2|ffn:d1p4t1e4)
```

This 16-GPU configuration uses PP=4, with attention modules using TP=2 and CP=2, while
expert modules use TP=1 and EP=4. See
[MoE Parallel Folding](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding)
for details on this feature.

### MUST KNOW: MoE Parallel Folding

MoE Parallel Folding is a Megatron-Core feature that **allows attention layers and expert
(FFN) layers within the same Transformer block to use different parallelism strategies on
the same set of GPUs**. Understanding this concept is essential for efficiently training
large MoE models.

#### Why It Matters

In a standard (non-folded) setup, all layers in a Transformer block share one parallelism
configuration. This creates a problem for MoE models:

- **Attention layers** benefit from Context Parallelism (CP) to handle long sequences, but
  do not have experts.
- **Expert FFN layers** benefit from Expert Parallelism (EP) to distribute experts across
  GPUs, but do not need CP.

Without folding, you would need `DP × TP × PP × CP × EP` GPUs—the product of *all*
dimensions—which quickly becomes prohibitive. For example, CP=2 and EP=4 alone would
require 8× more GPUs than a single replica.

Parallel folding solves this by **reusing the same GPU mesh for both module types with
different dimension mappings**, so the total GPU count is determined by the *larger* of the
two configurations, not their product.

#### How It Works

With the hybrid syntax, you specify independent parallelism for attention (`attn`) and FFN
(`ffn`) modules:

```
megatron:(attn:<attn_dims>|ffn:<ffn_dims>)
```

The GPUs allocated to CP in the attention section are **repurposed** for EP (and/or
different TP) in the FFN section. Concretely:

| Syntax                                         | Attention GPUs      | FFN GPUs            | Total GPUs |
| ---------------------------------------------- | ------------------- | ------------------- | ---------- |
| `megatron:(attn:d1p4t2c2\|ffn:d1p4t1e4)`      | DP=1, PP=4, TP=2, CP=2 | DP=1, PP=4, TP=1, EP=4 | **16**     |
| Without folding: `megatron:d1p4t2c2e4`         | —                   | —                   | **64**     |

By folding, the 16-GPU mesh serves **both** roles—CP=2 for attention and EP=4 for
experts—saving 4× the GPU count compared to the naive approach.

#### Key Rules

1. **Pipeline parallelism (`p`) must match** in both `attn` and `ffn` sections. If `p` is
   omitted from `ffn`, it inherits from `attn`.
2. **World sizes must be equal**: `attn_dp × attn_tp × attn_pp × attn_cp` must equal
   `ffn_dp × ffn_tp × ffn_pp × ffn_ep`.
3. **Context parallel (`c`) is only valid in `attn`**—it applies exclusively to attention
   modules.
4. **Expert parallel (`e`) is only valid in `ffn`**—it applies exclusively to expert
   modules.
5. **`ffn` data parallel (`d`) is auto-derived** if omitted:
   `ffn_dp = world_size / (ffn_ep × ffn_tp × ffn_pp)`.

#### Practical Example: Qwen3 30B-A3B on 32 GPUs

```
megatron:(attn:d1p4t2c2|ffn:d1p4t1e4)
```

- **Attention**: 4 pipeline stages, each with TP=2 and CP=2 → 1 × 4 × 2 × 2 = **16
  GPUs**
- **FFN/Experts**: 4 pipeline stages, each with TP=1 and EP=4 → 1 × 4 × 1 × 4 = **16
  GPUs**
- Same 16 GPUs are reused for both; with DP=2 replicas: **32 GPUs total**

```bash
actor.backend=megatron:(attn:d2p4t2c2|ffn:d2p4t1e4)  # 32 GPUs
```

#### When to Use Parallel Folding

- Your model is MoE **and** you need Context Parallelism for long sequences.
- You want to use Expert Parallelism without multiplying GPU requirements by `CP × EP`.
- You need fine-grained control over TP for attention vs. expert modules (expert layers
  often need less TP than attention layers).

**Tuning Guides:**

- [Megatron Performance Best Practice](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#performance-best-practice)
- [verl with Megatron Practice](https://github.com/ISEEKYAN/verl_megatron_practice)

## Aligning Inference and Training Precision

Due to the sparse nature of MoE models, the logits calculated by forward passes during
inference and training could be severely misaligned, leading to unstable training
results. To mitigate this instability, it is highly recommended to set
`actor.megatron.use_deterministic_algorithms=True` to disable nondeterministic
calculations in Megatron, although this may cause a ~10-20% slowdown in training steps.

As an example, you can run GRPO on the Qwen3 30B-A3B MoE model and GSM8K dataset (on a
32-GPU ray cluster) directly with the following command:

```bash
# NOTE: Allocation mode here is only for illustration purposes. It is not optimized.
python3 examples/math/gsm8k_rl.py --config <megatron_config.yaml> \
    scheduler.type=ray \
    experiment_name=megatron-moe-gsm8k-grpo trial_name=trial-0 \
    rollout.backend=sglang:d4t4 actor.backend=megatron:(attn:d1p4t2c2|ffn:d1p4t1e4) \
    cluster.n_nodes=4 cluster.n_gpus_per_node=8 actor.path=Qwen/Qwen3-30B-A3B \
    actor.megatron.use_deterministic_algorithms=True
```
