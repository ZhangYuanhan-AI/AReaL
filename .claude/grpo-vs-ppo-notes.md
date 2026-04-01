# GRPO vs PPO — RL Algorithm Study Notes

Deep dive into how GRPO and PPO compute advantages, the role of the critic model,
and temporal credit assignment.

---

## 1. Advantage: The Core Concept

```
advantage = "how much BETTER was this action than expected?"
          = actual outcome - baseline expectation
```

- advantage > 0 → "better than expected → increase probability"
- advantage < 0 → "worse than expected → decrease probability"
- advantage = 0 → "exactly as expected → no update"

### Three Baselines (Three Algorithms)

| Algorithm | Baseline | Advantage Formula |
|-----------|----------|-------------------|
| REINFORCE | None | advantage = reward |
| GRPO | Group of sequences | advantage = (reward - group_mean) / group_std |
| PPO | Learned critic V(t) | advantage(t) = GAE using δ(t) = r(t) + γ·V(t+1) - V(t) |

---

## 2. GRPO: Sequence-Level, Same Advantage for All Tokens

Task reward is a single number per sequence. It is **broadcast to every token equally**.
No GAE, no per-token decomposition.

```
Prompt: "What is 2+2?"
  Sequence A: [The] [answer] [is] [4]  → reward = 1.0
  Sequence B: [I]   [think]  [5]       → reward = 0.0
  Sequence C: [It's] [4]               → reward = 1.0
  Sequence D: [Maybe] [3]  [?]         → reward = 0.0

  group_mean = 0.5, group_std = 0.5

  Sequence A: advantage = (1.0 - 0.5) / 0.5 = +1.0
    → [The]=+1.0  [answer]=+1.0  [is]=+1.0  [4]=+1.0    ← all same

  Sequence B: advantage = (0.0 - 0.5) / 0.5 = -1.0
    → [I]=-1.0    [think]=-1.0   [5]=-1.0                ← all same
```

GRPO says: "this whole sequence was good/bad, reinforce/suppress everything equally."

### Why GRPO Works Despite Uniform Advantage

The advantage is the same for all tokens, but the **gradient is NOT the same**:

```
loss(t) = -advantage(t) × ratio(t)
ratio(t) = π_new(token_t) / π_old(token_t)
```

- `advantage` is uniform, but `ratio(t)` differs per token
- `∇ log π(token_t)` differs per token — rare/surprising tokens have steeper gradients
- The model naturally learns more from decisive tokens

GRPO also compensates with **many samples per prompt** (`n_samples: 4-16`):

```
Over many samples, the model observes:
  "[4]" appears in good sequences → reinforce
  "[5]" appears in bad sequences → suppress
```

Credit assignment happens **statistically across sequences**, not within a single one.

---

## 3. PPO: Per-Token Advantage via Critic (Temporal Credit Assignment)

### The Critic Model

The critic uses the **same pretrained Transformer body** as the actor, but replaces
`lm_head` (hidden → vocab_size) with `score_head` (hidden → 1). The score_head is
**randomly initialized** and trained from scratch during PPO training.

```python
# fsdp_engine.py:
if config.is_critic:
    model_class = AutoModelForTokenClassification  # score_head (hidden → 1)
else:
    model_class = AutoModelForCausalLM             # lm_head (hidden → vocab_size)
```

### What V(t) Means

V(t) = "Given tokens 0..t so far, how much total reward do I expect this sequence
to eventually get?"

```
Sequence: [What] [is] [2+2] [?] [The] [answer] [is] [4]  [EOS]

V(t):      0.5   0.5   0.5  0.5  0.5    0.6    0.7  1.0   0.0

Meaning:  "simple math          "starting   "almost  "correct!" "done"
           question, ~50%        to look      there"
           chance correct"       right"
```

### How PPO Computes Per-Token Advantage

PPO places the task reward at one token position, then uses the critic to decompose
it into per-token credit.

**Step 1: Construct per-token reward**

```
Sequence: [The] [answer] [is]  [4]   [EOS]
r(t):       0      0      0   +1.0     0     ← task reward at second-to-last token
                                               (KL penalty also added if kl_ctl > 0)
```

**Step 2: Compute TD error δ(t) at each token**

```
δ(t) = r(t) + γ·V(t+1) - V(t)

"What actually happened in this one step" minus "what critic expected"

V(t):    0.3    0.5     0.6    0.8    0.9

δ(0) = 0   + 1.0·0.5 - 0.3 = +0.2   "generating 'The' improved situation by 0.2"
δ(1) = 0   + 1.0·0.6 - 0.5 = +0.1   "generating 'answer' improved slightly"
δ(2) = 0   + 1.0·0.8 - 0.6 = +0.2   "generating 'is' improved situation"
δ(3) = 1.0 + 1.0·0   - 0.8 = +0.2   "generating '4' exceeded expectation by 0.2"
```

Each δ(t) is **different** because V(t) is different at each position.

**Step 3: GAE backward accumulation**

```
GAE(t) = δ(t) + γλ·GAE(t+1)

"Accumulate TD errors from the future, decayed by γλ"

With γ=1.0, λ=0.95:
  GAE(4) = 0
  GAE(3) = 0.2  + 0.95·0    = 0.20
  GAE(2) = 0.2  + 0.95·0.20 = 0.39
  GAE(1) = 0.1  + 0.95·0.39 = 0.47
  GAE(0) = 0.2  + 0.95·0.47 = 0.65

Advantage: [The]=0.65  [answer]=0.47  [is]=0.39  [4]=0.20  ← all different!
```

### PPO's Blame Assignment: Wrong Answer Example

```
Wrong: [The] [answer] [is] [5]  → reward = 0.0
V(t):   0.3    0.5     0.6  0.8  ← critic predicts same (hasn't seen final token yet)

δ(0) = 0   + 0.5 - 0.3 = +0.2    "seemed fine"
δ(1) = 0   + 0.6 - 0.5 = +0.1    "seemed fine"
δ(2) = 0   + 0.8 - 0.6 = +0.2    "seemed fine"
δ(3) = 0.0 + 0   - 0.8 = -0.8    "'5' was TERRIBLE — expected 0.8, got 0"
                   ^^^^
                   THIS is where PPO assigns blame

GAE (γ=1, λ=0.95):
  GAE(3) = -0.80                          ← strong negative
  GAE(2) = 0.2 + 0.95·(-0.80)  = -0.56   ← blame propagates back
  GAE(1) = 0.1 + 0.95·(-0.56)  = -0.43   ← less blame
  GAE(0) = 0.2 + 0.95·(-0.43)  = -0.21   ← even less

PPO: "token '5' was the problem (-0.80). Earlier tokens were less to blame."
GRPO: "everything was equally bad (-1.0 each)."
```

---

## 4. GAE Formula Explained

### Why Not Just Use δ(t)?

```
δ(t) alone:    low variance, high bias   (trusts critic, which may be wrong)
Full returns:  low bias, high variance    (no critic trust, but noisy)
GAE:           blends both via λ
```

### The Recursive Formula

```
GAE(t) = δ(t) + γλ·δ(t+1) + (γλ)²·δ(t+2) + ...
       = δ(t) + γλ·GAE(t+1)                          ← recursive form

λ controls how far back the reward signal travels:
  λ=0:    each token only sees its own one-step δ(t)
  λ=0.95: each token sees the whole future, decayed by 0.95 per step
  λ=1:    each token sees the full undiscounted future
```

### The Backward Loop (Implementation)

```python
lastgaelam = 0
for t in reversed(range(max_seqlen - 1)):       # t = T-1, ..., 1, 0
    delta = rewards[t] + γ * values[t+1] - values[t]
    lastgaelam = delta + γ * λ * lastgaelam
    advantages[t] = lastgaelam
```

One pass, O(n), computes the recursive formula from right to left.

---

## 5. Actor and Critic: Trained Together

In PPO, actor and critic are trained **in the same loop, as separate models with
separate optimizers**:

```
Each PPO step:
  1. Actor generates sequences (rollout)
  2. Critic computes V(t) for each token
  3. Compute advantages using δ(t) and GAE
  4. Update actor: policy gradient loss using advantages    ← actor optimizer
  5. Update critic: (V(t) - actual_return)² regression     ← critic optimizer
```

They depend on each other:
- Actor needs critic: advantages depend on V(t) estimates
- Critic needs actor: V(t) targets come from actor's rollout rewards

---

## 6. How to Use PPO in AReaL

Add a `critic:` section to the YAML config. No code changes needed.

```yaml
critic:
  backend: ${actor.backend}
  is_critic: true                    # ← this swaps lm_head → score_head
  path: ${actor.path}               # same pretrained model, head replaced
  optimizer: ${actor.optimizer}      # critic has its own optimizer
  scheduling_strategy:
    type: colocation
    target: actor                    # share GPUs with actor
```

---

## 7. The Trade-Off

| | GRPO | PPO |
|---|---|---|
| Models to train | 1 (actor) | 2 (actor + critic) |
| GPU memory | Less | More (extra model) |
| Advantage per token | Same (broadcast) | Different (per-token credit) |
| Credit assignment | Statistical (across sequences) | Temporal (within sequence) |
| Advantage quality at start | Decent (group stats) | Poor (random score_head) |
| Convergence | Fast (no critic warmup) | Slower (critic must learn first) |
| Needs many samples | Yes (n_samples=4-16) | Less critical |
| When to use | Most tasks | When per-token credit matters |

### The Fundamental Difference

```
GRPO:  "Was this SEQUENCE good?" → one answer, broadcast to all tokens
       Credit assigned statistically across many sequences.

PPO:   "Was this TOKEN a good action?" → different answer per token
       Credit assigned temporally within a single sequence via critic.
```

---

## 8. In AReaL Code

Both use the same `_compute_advantages()` in `areal/trainer/ppo/actor.py`.
The only branching point:

```python
if "values" not in data:
    values = torch.zeros_like(rewards)    # ← GRPO: no critic, V=0
else:
    values = data["values"]               # ← PPO: critic provides V(t)
```

With V=0, γ=1, λ=1, kl_ctl=0 (GRPO defaults): GAE degenerates into "total future
reward" which is the same task reward at every token → uniform advantage.

The critic V(t) is what breaks this symmetry and enables per-token advantages.

Loss function is also shared (`ppo_actor_loss_fn` in `areal/utils/functional/functional.py`):

```python
ratio = exp(logprobs_new - logprobs_old)        # per-token probability ratio
clipped_ratio = clip(ratio, 1-eps, 1+eps)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

Same formula, same code. Only the advantage values differ between GRPO and PPO.
