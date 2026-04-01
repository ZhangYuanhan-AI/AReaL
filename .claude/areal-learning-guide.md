# AReaL Technical Learning Guide — Systematic Overview

A structured roadmap covering every technical concept in the AReaL codebase,
organized from foundational to advanced.

---

## Part 1: Architecture — How AReaL Works

### Single-Controller Design

AReaL uses a **single controller process** that orchestrates everything via RPC, unlike
traditional SPMD (Single Program Multiple Data) frameworks where every GPU runs the same
program.

```
┌──────────────────────────────────────────────────┐
│                  Controller                       │
│  (Python, single thread, orchestrates all)        │
│                                                   │
│  1. Send prompts to rollout workers               │
│  2. Collect generated sequences                   │
│  3. Dispatch to training workers                  │
│  4. Trigger weight sync                           │
└──────┬──────────┬──────────┬──────────┬──────────┘
       │ RPC      │ RPC      │ RPC      │ RPC
       ▼          ▼          ▼          ▼
   [Rollout]  [Rollout]  [Train]    [Train]
   SGLang/    SGLang/    FSDP/      FSDP/
   vLLM       vLLM       Megatron   Megatron
```

**Key files:**
- `areal/infra/controller/` — Controller logic
- `areal/infra/rpc/` — RPC communication layer
- `areal/infra/scheduler/` — Resource scheduling (Local, Ray, SLURM)

### Data Flow

```
Prompts → [Rollout Engine] → Generated sequences
                                    ↓
                            [Reward Function] → scores
                                    ↓
                            [Training Engine] → updated weights
                                    ↓
                            [Weight Sync] → rollout loads new weights
                                    ↓
                              Next iteration
```

### Key Abstraction: RTensor

RTensor is AReaL's distributed tensor abstraction — it handles transferring tensor data
between rollout and training workers across machines without you managing serialization.

---

## Part 2: RL Algorithms

### The GRPO Family

All algorithms share the same code path with different config switches:

| Algorithm | Key Difference | Config Switch |
|-----------|---------------|---------------|
| **GRPO** | Group-relative advantage, no critic needed | `adv_norm=group` |
| **Dr.GRPO** | GRPO with variance-reduced baseline | `reward_norm=dr_grpo` |
| **DAPO** | Dynamic sampling, no clipping | `eps_clip=null, dynamic_sampling=True` |
| **PPO** | Classic policy gradient with critic | Requires critic engine |
| **LitePPO** | PPO with frozen critic (saves memory) | `freeze_critic=True` |
| **RLOO** | Leave-one-out baseline | `adv_norm=rloo` |
| **GSPO** | Group-sorted advantage | `adv_norm=gspo` |
| **SAPO** | Self-adaptive, combines SFT + RL signal | `sapo.enabled=True` |
| **M2PO** | Mixed maximize-minimize policy optimization | `m2po.enabled=True` |

**Key files:**
- `areal/utils/functional.py` — Core RL math (advantage, clipping, loss)
- `docs/en/algorithms/grpo_series.md` — Algorithm family documentation
- `docs/en/algorithms/async.md` — Asynchronous training with staleness control
- `docs/en/algorithms/distillation.md` — Knowledge distillation (teacher model)

### Core RL Math

```python
# Simplified GRPO loss
advantage = (reward - group_mean) / group_std          # group-relative
ratio = exp(log_prob_new - log_prob_old)                # importance sampling
clipped_ratio = clip(ratio, 1-eps, 1+eps)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

**What you should know:**
- Reward normalization level (none, batch, group, dr_grpo)
- Advantage normalization level (none, group, rloo, gspo)
- KL penalty (controls how far new policy drifts from reference)
- Importance sampling (off-policy correction when training is async)

---

## Part 3: Training Engines

### Three Backends

| Engine | Tech | Parallelism | Best For |
|--------|------|-------------|----------|
| **FSDP2** | PyTorch native FSDP | DP, TP, CP | Simple setups, fast iteration |
| **Megatron** | NVIDIA Megatron-Core | DP, TP, PP, CP, EP | Large MoE, production scale |
| **Archon** | PyTorch-native (experimental) | DP, TP, PP, CP, EP | New projects, torch.compile |

**Key files:**
- `areal/engine/fsdp_engine.py` — FSDP2 engine
- `areal/engine/megatron_engine.py` — Megatron engine
- `areal/experimental/engine/archon_engine.py` — Archon engine

### FSDP2 (Fully Sharded Data Parallel)

Shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds
a shard; parameters are gathered on-demand for forward/backward, then resharded.

```
GPU 0: shard 0 of all params  →  gather full param → compute → discard → keep shard
GPU 1: shard 1 of all params  →  gather full param → compute → discard → keep shard
```

**Key concepts:**
- Parameter sharding (ZeRO Stage 3)
- Mixed precision (bf16 compute, fp32 master weights)
- Gradient accumulation for effective large batch sizes
- `areal/engine/fsdp_utils/` — Checkpoint, gradient, optimizer, parallel utilities

### Megatron Engine

NVIDIA's Megatron-Core with full 5D parallelism. See our previous discussion on
TP/PP/DP/CP/EP and MoE Parallel Folding.

**Key concepts:**
- 5D parallelism configuration via allocation mode strings
- MoE Parallel Folding (hybrid attn/ffn syntax)
- Pipeline schedules (1F1B, interleaved)
- FP8 quantization support
- `areal/engine/megatron_utils/` — Pipeline parallel, checkpoint, quantization

### Archon Engine (Experimental)

Pure PyTorch implementation that achieves Megatron-level parallelism without the
Megatron dependency. Supports torch.compile for kernel fusion.

**Key concepts:**
- Same 5D parallelism as Megatron
- Simpler codebase, easier to extend
- Memory tuning for pipeline parallelism (warmup accumulation, zero-bubble)
- `areal/experimental/engine/archon_engine.py`
- `areal/experimental/models/archon/` — Model architecture adaptations

---

## Part 4: Inference Engines

### SGLang (Recommended)

High-performance inference server with continuous batching. AReaL wraps it as a
subprocess.

```yaml
rollout:
  backend: "sglang:d4t4"   # 4 instances × 4 TP GPUs = 16 GPUs
```

**Key concepts:**
- Continuous batching (dynamic batch assembly)
- DP attention (data parallel across instances)
- Expert parallelism within each instance
- `areal/engine/sglang_engine.py`

### vLLM

Alternative inference backend with PagedAttention.

```yaml
rollout:
  backend: "vllm:d2t4"
```

- `areal/engine/vllm_engine.py`

---

## Part 5: Parallelism Deep Dive

### The 5 Dimensions (Recap)

```
Total GPUs = DP × TP × PP × CP
EP reuses existing GPUs (parallel folding)
```

| Dim | Splits | Saves | Communication | Constraint |
|-----|--------|-------|---------------|------------|
| DP | Data batches | Nothing (adds throughput) | Gradient all-reduce (infrequent) | None |
| TP | Weight matrices | Model memory per GPU | All-reduce per operation (frequent) | TP ≤ GPUs per node |
| PP | Model layers | Model memory per GPU | Activation transfer between stages | Pipeline bubbles |
| CP | Sequence length | Activation memory (O(n²)) | Ring attention | Only helps attention |
| EP | MoE experts | Expert weight memory | All-to-all token routing | Only helps MoE FFN |

### Allocation Mode System

```
# Simple
actor.backend: "fsdp:d8"
actor.backend: "megatron:d2p2t4"

# MoE hybrid (parallel folding)
actor.backend: "megatron:(attn:d2p4t2c2|ffn:d2p4t1e4)"

# Inference
rollout.backend: "sglang:d4t4"
```

**Key file:** `areal/api/alloc_mode.py` — Grammar parser, validation, strategy construction

---

## Part 6: Workflows

### RolloutWorkflow: The Core Abstraction

A workflow defines the loop: generate → evaluate → train. Different workflows handle
different interaction patterns.

| Workflow | Use Case | Key File |
|----------|----------|----------|
| **RLVRWorkflow** | Single-turn Q&A (math, coding) | `areal/workflow/rlvr.py` |
| **MultiTurnWorkflow** | Multi-turn agent conversations | `areal/workflow/multi_turn.py` |
| **VisionRLVRWorkflow** | Vision-language models | `areal/workflow/vision_rlvr.py` |

### Agentic RL Workflows

For training agents that use tools, APIs, and multi-step reasoning:

```
Prompt → Agent generates action → Environment responds → Agent continues → ... → Reward
```

Two integration approaches:
- **Proxy approach** (recommended): HTTP proxy captures all LLM calls transparently
- **Direct approach**: Custom ArealOpenAI client

**Key files:**
- `docs/en/tutorial/agentic_rl.md` — Agentic RL tutorial
- `docs/en/customization/agent.md` — Custom agent workflow guide
- `examples/agent/` — Agent examples (OpenAI Agents, CAMEL, LangChain, Claude SDK)

---

## Part 7: Rewards

### Built-in Reward Functions

| Reward | Domain | Key File |
|--------|--------|----------|
| Math reward | GSM8K, MATH | `areal/reward/math.py` |
| Code reward | HumanEval, coding | `areal/reward/code.py` |
| Geometry3K | Geometry problems | `areal/reward/geometry3k.py` |
| Countdown | Number game | `areal/reward/countdown.py` |
| Format reward | Output format compliance | `areal/reward/format.py` |

### Custom Rewards

Implement the reward API interface:

```python
# areal/api/reward_api.py defines the contract
def my_reward(response: str, ground_truth: str) -> float:
    ...
```

---

## Part 8: Datasets

### Built-in Loaders

| Dataset | Domain | Key File |
|---------|--------|----------|
| GSM8K | Grade school math | `areal/dataset/gsm8k.py` |
| MATH | Competition math | `areal/dataset/math.py` |
| Geometry3K | Geometry | `areal/dataset/geometry3k.py` |
| Countdown | Number game | `areal/dataset/countdown.py` |
| HuggingFace | Generic HF datasets | `areal/dataset/huggingface.py` |

### Custom Datasets

Must provide `messages` (prompt) and `answer` (ground truth) fields. See
`docs/en/customization/dataset.md`.

---

## Part 9: Infrastructure

### Schedulers

| Scheduler | Use Case | Key File |
|-----------|----------|----------|
| **Local** | Single node, debugging | `areal/infra/scheduler/local.py` |
| **Ray** | Multi-node clusters | `areal/infra/scheduler/ray.py` |
| **SLURM** | HPC clusters | `areal/infra/scheduler/slurm.py` |

### Launchers

Handle process spawning on each node:

- `areal/infra/utils/launcher.py` — Process launcher
- `areal/infra/utils/proc.py` — Process management

### RPC Layer

Remote procedure calls between controller and workers:

- `areal/infra/rpc/` — RPC server/client implementation

---

## Part 10: Checkpointing

### Two Complementary Systems

| System | Purpose | Format | Key File |
|--------|---------|--------|----------|
| **Saver** | Export for evaluation/deployment | HuggingFace format | `areal/engine/core/checkpoint.py` |
| **RecoverHandler** | Fault tolerance during training | PyTorch DCP | `areal/engine/core/checkpoint.py` |

- HF checkpoints: loadable by `transformers.AutoModel`
- DCP checkpoints: distributed checkpoint protocol, handles resharding across different
  GPU counts

---

## Part 11: Performance & Optimization

### Key Concepts

- **Async training**: Overlap rollout and training for better GPU utilization
  (`docs/en/algorithms/async.md`)
- **Staleness control**: Importance sampling correction for off-policy data
- **Micro-batching**: `mb_spec.max_tokens_per_mb` controls memory per micro-batch
- **Concurrent rollouts**: `max_concurrent_rollouts` — most impactful OOM knob
- **Gradient accumulation**: Effective batch size = micro_batch × accumulation_steps × DP

### Profiling

Chrome Trace-compatible profiling for visualization in Perfetto:
- `docs/en/best_practices/perf_profiling.md`

### Common Issues

| Problem | First Thing to Try |
|---------|--------------------|
| Training OOM | Reduce `mb_spec.max_tokens_per_mb` |
| Generation OOM | Reduce `max_concurrent_rollouts` |
| Rewards not increasing | Check `docs/en/best_practices/algo_perf.md` |
| Training instability (MoE) | Set `use_deterministic_algorithms=True` |

---

## Part 12: FP8 & Quantization

Megatron engine supports FP8 training for reduced memory and faster compute:

- `areal/engine/megatron_utils/quantization.py` — FP8 quantization utilities
- Requires NVIDIA H100+ GPUs with FP8 hardware support

---

## Part 13: Monitoring & Metrics

### Two Metric Paradigms

| Type | Source | Timing |
|------|--------|--------|
| **Streaming** | Rollout workers | Continuous during generation |
| **Batch** | Training workers | After each training step |

Key metrics to watch:
- `reward/mean` — Is the model improving?
- `policy/importance_weight` — Is off-policy correction working?
- `loss/policy_loss` — Is training stable?
- `throughput/tokens_per_second` — Is the system efficient?

**Key file:** `docs/en/best_practices/metrics_tracking.md`

---

## Part 14: Extending AReaL

### Difficulty Levels

| Task | Difficulty | Guide |
|------|-----------|-------|
| Add a dataset | Easy | `/add-dataset` skill, `docs/en/customization/dataset.md` |
| Add a reward function | Easy | `/add-reward` skill, `areal/api/reward_api.py` |
| Add a workflow | Moderate | `/add-workflow` skill, `docs/en/customization/agent.md` |
| Add an Archon model | Moderate | `/add-archon-model` skill |
| Modify an engine | Hard | Requires deep distributed systems knowledge |
| Modify scheduler/launcher | Hard | Requires cluster management knowledge |

---

## Recommended Learning Order

```
Week 1: Foundation
├── 1. Architecture (Part 1) — understand the controller + worker model
├── 2. Run quickstart — docs/en/tutorial/quickstart.md
├── 3. RL Algorithms overview (Part 2) — GRPO is the default, understand it
└── 4. Allocation modes (Part 5) — know how to read d2p4t2 strings

Week 2: Engines & Parallelism
├── 5. Choose your engine (Part 3) — FSDP for simple, Megatron for MoE
├── 6. Parallelism deep dive (Part 5) — TP/PP/CP/EP and when to use each
├── 7. MoE Parallel Folding — our previous discussion notes
└── 8. Inference engines (Part 4) — SGLang setup

Week 3: Workflows & Customization
├── 9. Workflows (Part 6) — RLVR vs multi-turn vs agent
├── 10. Agentic RL — docs/en/tutorial/agentic_rl.md
├── 11. Rewards & Datasets (Parts 7-8) — built-in and custom
└── 12. Try examples/ — run a real training job

Week 4: Production & Debugging
├── 13. Performance tuning (Part 11) — async, micro-batching, OOM
├── 14. Checkpointing (Part 10) — HF export vs DCP recovery
├── 15. Monitoring (Part 13) — what metrics to watch
└── 16. Infrastructure (Part 9) — Ray/SLURM for multi-node
```

---

## Quick Reference: "I want to do X"

| I want to... | Start here |
|--------------|-----------|
| Train a math reasoning model | `examples/math/gsm8k_rl.py` |
| Train an agent with tools | `docs/en/tutorial/agentic_rl.md` |
| Use a bigger model that doesn't fit on 1 GPU | Increase TP, then PP (Part 5) |
| Train an MoE model | `docs/en/tutorial/megatron.md` |
| Train on a vision-language model | `examples/vision/` |
| Add my own dataset | `/add-dataset` skill |
| Add my own reward | `/add-reward` skill |
| Debug OOM errors | `docs/en/best_practices/handling_oom.md` |
| Debug training not converging | `docs/en/best_practices/algo_perf.md` |
| Deploy on a SLURM cluster | `docs/en/tutorial/quickstart.md` (distributed section) |
| Deploy on cloud (AWS/GCP) | `examples/skypilot/` |
