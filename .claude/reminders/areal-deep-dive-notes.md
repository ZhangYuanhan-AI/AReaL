# AReaL Deep Dive Notes

Key takeaways from a systematic exploration of the AReaL codebase.

---

## 1. Overall Architecture

AReaL is a distributed RL training framework with three pillars:

- **Training** (sync): FSDP or Megatron engines compute PPO/GRPO loss and update weights
- **Rollout** (async): SGLang or vLLM engines generate completions on separate GPUs
- **Coordination**: Scheduler launches workers, WorkflowExecutor runs workflows
  asynchronously, StalenessManager tracks version drift

The magic: async rollout + sync training = high GPU utilization. While rollout GPUs
generate new trajectories, training GPUs compute updates on previous trajectories.

### Entry Point Flow

```
examples/math/gsm8k_rl.py
  → load_expr_config(GRPOConfig)
  → PPOTrainer(config, train_dataset, valid_dataset)
  → trainer.train(workflow="RLVRWorkflow", workflow_kwargs=...)
```

### Main Training Loop (per step)

1. **Rollout**: `actor.prepare_batch()` → WorkflowExecutor → `arun_episode()` × N
2. **Compute values**: critic estimates (if PPO, not GRPO)
3. **Compute ref logprobs**: reference policy (for KL penalty, if kl_ctl > 0)
4. **Compute advantages**: group-level reward normalization for GRPO
5. **PPO update**: clipped surrogate loss, backward, optimizer step
6. **Pause rollout** → weight sync (NCCL) → save/eval → **resume rollout**

---

## 2. `areal/api/` vs `areal/engine/` — Contract vs Implementation

`areal/api/` is the **public programming surface** — ABCs, data types, configs. The name
"API" means Application Programming Interface (not HTTP API): the set of classes and
types you program against.

`areal/engine/` contains **concrete implementations** that inherit from the API.

```
api/engine_api.py  →  TrainEngine (ABC)      ←  engine/fsdp_engine.py: FSDPEngine
                   →  InferenceEngine (ABC)   ←  engine/sglang_remote.py: RemoteSGLangEngine
api/workflow_api.py → RolloutWorkflow (ABC)   ←  workflow/rlvr.py: RLVRWorkflow
api/scheduler_api.py → Scheduler (ABC)        ←  infra/scheduler/local.py: LocalScheduler
```

**Key rule**: Workflow authors import from `areal/api/` only, never from `areal/engine/`.
This is dependency inversion — high-level logic depends on abstractions, not
implementations.

### api/ file catalog

| File              | Contents                                                 |
| ----------------- | -------------------------------------------------------- |
| `engine_api.py`   | `TrainEngine` + `InferenceEngine` ABCs                   |
| `workflow_api.py`  | `RolloutWorkflow` ABC                                    |
| `io_struct.py`    | Data types: `ModelRequest`, `ModelResponse`, `WeightUpdateMeta` |
| `cli_args.py`     | Config dataclasses: `GRPOConfig`, `PPOActorConfig`       |
| `scheduler_api.py` | `Scheduler` ABC                                         |
| `alloc_mode.py`   | `ModelAllocation`, `ParallelStrategy`                    |
| `reward_api.py`   | `AsyncRewardWrapper`                                     |

---

## 3. GSM8K GRPO Example — The Baseline

**Single-turn**: prompt → one generation → reward → done.

- Model: Qwen2.5-1.5B-Instruct
- Workflow: built-in `RLVRWorkflow`
- Config: `fsdp:d4p1t1` (4 GPUs training), `sglang:d4p1t1` (4 GPUs inference)
- GRPO: `n_samples=4` per prompt, group-level normalization, `kl_ctl=0.0` (no ref model)
- Reward: `math_verify` checks `\boxed{}` against ground truth → 0.0 or 1.0

---

## 4. TIR Example — Multi-Turn with Tools (Manual Token Control)

**Multi-turn**: prompt → [generate → detect tool marker → execute tool → inject result]*
→ reward.

Key design decisions:

- **loss_mask=1** on model-generated tokens (including tool markers like `` ```python ``)
- **loss_mask=0** on tool output (model doesn't learn to predict tool results)
- **State machine** with two phases: stop at start markers, then stop at end markers
- **Tools**: PythonTool (subprocess execution), CalculatorTool (safe eval)
- Model learns **when** to call tools through RL — no supervised tool-use data

The commented-out reward bonus (`tool_using = 0.01`) was deliberately removed — pure
outcome reward lets the model discover tool usage organically.

---

## 5. Multi-Turn Math — ArealOpenAI Abstraction

**Message-level control**: uses `ArealOpenAI` client that looks like OpenAI's SDK but
records token-level data automatically.

```python
client = ArealOpenAI(engine=engine, tokenizer=tokenizer)
response = await client.chat.completions.create(messages=messages)
client.set_reward(response.id, reward)
client.apply_reward_discount(turn_discount=0.9)
return client.export_interactions(style="concat")
```

No manual tensor management. The client handles `input_ids`, `logprobs`, `loss_mask`,
`versions` invisibly.

---

## 6. "Same Code for Training and Evaluation"

The `RolloutWorkflow.arun_episode(engine, data)` interface is agnostic to whether it's
called during training or evaluation. Both paths use the same workflow class — only
`temperature` differs (1.0 for training exploration, 0.6 for evaluation).

This matters most for complex workflows like TIR where multi-turn tool-calling logic
would inevitably diverge if written twice.

---

## 7. Token-Level vs Message-Level Control

Two paradigms for multi-turn workflows:

### Token-level (TIR manual approach)

- You build `seq`, `logprobs`, `loss_mask`, `versions` lists manually
- Tool markers (`` ```python ``, `` ``` ``) stay in the sequence with `loss_mask=1`
- The model gets gradient signal on producing tool markers themselves
- Full control over every token

### Message-level (ArealOpenAI)

- Each `chat.completions.create()` is a turn
- `InteractionCache` records everything automatically
- `export_interactions()` produces training tensors
- Two export styles:
  - `concat`: builds conversation tree, returns leaf with all turns stitched into one
    sequence
  - `individual`: returns each turn as a separate trajectory

### ArealOpenAI Concat Limitations for Tool-Calling

1. **Stop tokens silently dropped** (line 486 in client.py): concat mode can't handle
   stop-token-truncated outputs because the EOS-counting alignment breaks
2. **Stop markers stripped**: `output_tokens_without_stop` is used for parent stitching,
   so `` ```python `` is replaced by EOS — the model never trains on producing the
   marker itself

### When to Use Which

| Decision boundary is...                                  | Use                  |
| -------------------------------------------------------- | -------------------- |
| Between messages (multi-turn chat, retry, reflection)    | ArealOpenAI          |
| Inside a single generation (tool markers, code blocks)   | Manual token control |
| Both (agent with tools AND multi-turn)                   | Manual, or fix ArealOpenAI |

---

## 8. Learning Path for Agentic RL

| Level | Example            | Key New Concept                                     |
| ----- | ------------------ | --------------------------------------------------- |
| 1     | `math/`            | Full system: config → train loop (built-in workflow) |
| 2     | `countdown/`       | Write your own `RolloutWorkflow` (single-turn)      |
| 3     | `tir/`             | Multi-turn generation + tool execution (manual)     |
| 4     | `multi_turn_math/` | `ArealOpenAI`: automatic token tracking             |
| 5     | `agent_workflow/`  | Proxy integration: any agent framework              |
| 6     | `openclaw/`        | Online RL: external agents, HTTP API                |

Each level adds one abstraction layer. The training loop (`PPOTrainer.train()`) never
changes — only what happens inside `arun_episode` evolves.

---

## 9. My Role in Agentic RL

(See `.claude/reminders/my-role-in-agentic-rl.md`)

Code is the easy part. The real work is:

1. **Choose what to build** — "Train tool use via RL, not SFT"
2. **Design the learning signal** — pure outcome reward, no tool-use bonus
3. **Decide what the model should learn** — loss_mask=1 for model tokens, 0 for tool
   output
4. **Set the experimental space** — max_turns, temperature, n_samples
5. **Run experiments and interpret results** — "15% accuracy advantage over pure GRPO"
6. **Identify what's missing** — "ArealOpenAI concat needs stop token support for TIR"
7. **Make architectural bets** — "Same workflow for train and eval, always"

Faster implementation means more ideas per week, making judgment and experimental
intuition *more* valuable, not less.