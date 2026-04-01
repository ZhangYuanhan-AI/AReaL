# MoE Parallel Folding & Distributed Training — Study Notes

Summary of discussion on parallelism dimensions, MoE routing, attention mechanics,
and FlashAttention.

---

## 1. The 5 Parallelism Dimensions

| Dim | Full Name | What It Splits | Why It Needs Dedicated GPUs |
|-----|-----------|---------------|---------------------------|
| **DP** | Data Parallel | Training data batches | Each replica processes different data simultaneously |
| **TP** | Tensor Parallel | Weight matrices (and attention heads) | Both halves compute at the same time |
| **PP** | Pipeline Parallel | Model layers into stages | Each stage holds different layers |
| **CP** | Context Parallel | Sequence length (attention only) | Each group processes different tokens simultaneously |
| **EP** | Expert Parallel | MoE experts across GPUs | Reuses existing GPUs — does NOT add to GPU count |
| **ETP** | Expert Tensor Parallel | TP within expert FFN layers specifically | Can differ from attention TP |

**GPU count formula**: `total = DP × TP × PP × CP`

EP does not multiply in because attention and FFN run sequentially, not simultaneously.
The same GPUs that do CP during attention are repurposed for EP during FFN.

---

## 2. MoE Parallel Folding

### The Problem

Without folding: `DP × TP × PP × CP × EP` GPUs needed — multiplicative explosion.
Example: CP=2, EP=4 → 8× more GPUs than needed.

### The Solution

Attention and FFN run sequentially within each Transformer block. CP is only useful
during attention (O(n²) cross-token interaction). EP is only useful during FFN (routing
tokens to experts). Folding reuses the same GPU mesh with different communication
patterns for each phase.

### AReaL Syntax

```
megatron:(attn:d2p4t2c2|ffn:d2p4t1e4)   # 32 GPUs
```

- Attention: DP=2, PP=4, TP=2, CP=2 → 2×4×2×2 = 32 GPUs
- FFN: DP=2, PP=4, TP=1, EP=4 → 2×4×1×4 = 32 GPUs (same GPUs, different grouping)

### Key Rules

1. PP must match in both attn and ffn sections
2. World sizes must be equal
3. CP only valid in attn section
4. EP only valid in ffn section
5. FFN dp is auto-derived if omitted

---

## 3. Why TP and PP Both Exist

Both save **model weight memory**, but with different trade-offs:

| | TP | PP |
|---|---|---|
| How | Splits each layer's weights across GPUs | Puts different layers on different GPUs |
| Communication | Frequent, small (all-reduce per operation) | Infrequent, larger (pass activations between stages) |
| Requires NVLink | Yes (must be within a node) | No (can cross nodes) |
| Idle time | None | Pipeline bubbles |

**CP** is the one that saves **activation memory** (the O(n²) attention scores).

---

## 4. Attention Calculation

### Core Math

For each token i, compute dot product with all keys, softmax, then weighted sum of values:

```
score(i, j) = dot(Q_i, K_j) / √d
weights[i] = softmax(scores[i, :])      ← one row
output[i]  = Σ weights[i,j] × V_j
```

n tokens → n×n score matrix → O(n²) compute and memory.

### Multi-Head Attention

Split into H independent heads, each with smaller dimension:

```
head_dim = hidden_dim / num_heads  (common convention, not hard rule)
```

Each head learns different attention patterns. Results are concatenated and projected
back via W_O.

### Weight Matrices in Attention

```
W_Q: (hidden_dim × hidden_dim)   maps hidden → queries
W_K: (hidden_dim × hidden_dim)   maps hidden → keys
W_V: (hidden_dim × hidden_dim)   maps hidden → values
W_O: (hidden_dim × hidden_dim)   maps concatenated heads → hidden

Total: 4 × hidden_dim² parameters
```

Modern models use GQA (Grouped Query Attention) where K/V have fewer heads than Q,
making W_K and W_V smaller.

### head_dim vs hidden_dim

- `hidden_dim = num_heads × head_dim` is convention, not a hard rule
- head_dim < hidden_dim (almost always — you have multiple heads)
- DeepSeek-V3 breaks this convention with MLA (Multi-head Latent Attention)
- Hard constraint: Q and K must share the same head_dim (they dot-product)

---

## 5. MoE Routing

### How Tokens Choose Experts

```
router_logits = x @ W_router       (hidden_dim × num_experts) — tiny matrix
top-K experts selected per token    (commonly K=2)
gate_weights = softmax over selected experts
output = Σ gate_k × Expert_k(x)
```

Router weight is negligible: e.g., 4096×128 = 1MB vs 6.4GB for all expert FFN weights.

### EP Communication Pattern

With EP, experts are distributed. Tokens are routed via all-to-all:
1. All-to-all: send tokens to GPUs holding their selected experts
2. Each GPU computes its local experts
3. All-to-all back: return results to original GPUs
4. Combine with gate weights

---

## 6. Why CP Helps Attention but Not FFN

```
FFN:        per-token operation, O(n) in sequence length    ← linear, manageable
Attention:  every token × every token, O(n²)                ← explodes with length

n=128K:
  FFN activations:     ~2.7 GB     ← fine
  Attention scores:    ~1 TB       ← impossible without optimization
```

CP splits the query tokens across GPUs, halving the O(n²) compute per GPU.
FFN doesn't need this because it processes each token independently.

---

## 7. FlashAttention

### Problem

Standard attention materializes the full n×n score matrix → O(n²) memory.

### Solution

Process Q×K in small blocks (B×B), never storing the full matrix. Use **online softmax**
to incrementally maintain correct normalization across blocks.

### Algorithm (Simplified)

For each Q block, iterate over all K/V blocks:
1. Compute small (B×B) score chunk
2. Update running max, running sum, running output using online softmax correction
3. Discard the score chunk
4. After all K blocks processed: normalize final output

### Key Properties

```
              Compute    Memory
Standard:     O(n²)      O(n²)
FlashAttention: O(n²)    O(n)     ← memory savings only
CP=2:         O(n²/2)   O(n²/2)  ← both halved
```

FlashAttention and CP are complementary:
- FlashAttention fixes the memory problem
- CP fixes the compute problem

---

## 8. Conceptual Insight: HP ≈ EP

"Head Parallelism" (splitting independent attention heads) is semantically the same
pattern as Expert Parallelism (splitting independent experts). Both distribute
independent compute units across GPUs. TP in attention layers is effectively "HP" —
it falls out naturally because slicing the Q/K/V projection matrix by columns
automatically gives per-head splits.

---

## 9. Practical Guidance for Agentic RL Training

**Must know**: what d/t/p/c/e mean, how to calculate GPU count, the hybrid syntax for
MoE folding, TP ≤ GPUs-per-node.

**Don't need to know**: internal implementation of any parallelism, process group
construction, gradient reduction algorithms, ring attention mechanics.

**Rules of thumb**:
- Model doesn't fit → increase TP (within node), then PP (across nodes)
- Want more throughput → increase DP
- Long sequences → add CP
- MoE with many experts → add EP
- MoE + long sequences → use parallel folding (hybrid syntax)
- MoE training instability → set `use_deterministic_algorithms=True`
