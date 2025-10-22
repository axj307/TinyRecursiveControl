# Tiny Recursive Models (TRM) - Paper Architecture

**Paper:** "Less is More: Recursive Reasoning with Tiny Networks"
**arXiv:** [2510.04871](https://arxiv.org/abs/2510.04871)
**Author:** Alexia Jolicoeur-Martineau
**Domain:** ARC-AGI puzzle solving

---

## Overview

Tiny Recursive Models (TRM) is a recursive reasoning approach that achieves 45% accuracy on ARC-AGI-1 and 8% on ARC-AGI-2 using only a **7M parameter neural network**. The core innovation is achieving strong reasoning performance through **recursive application of tiny networks** rather than scaling up model size.

### Key Achievement
- **7M parameters** vs billions in LLMs
- **45% on ARC-AGI-1** (competitive with LLM approaches)
- **Parameter efficiency** through weight sharing
- **Recursive reasoning** enables iterative improvement

---

## Core Architecture

### High-Level Flow

```
Input: Discrete grid tokens [batch, seq_len]
    ↓
[1] Embeddings (tokens + puzzle + position)
    ↓
[2] Initialize latent states: z_H, z_L
    ↓
[3] Recursive Refinement (H_cycles iterations)
    For H_step in range(H_cycles):  # e.g., 3
        For L_step in range(L_cycles):  # e.g., 4-6
            z_L = L_level(z_L, z_H + input)  ← Low-level reasoning
        z_H = L_level(z_H, z_L)  ← High-level reasoning
    ↓
[4] Adaptive Computation Time (ACT)
    Q-head decides whether to halt
    ↓
[5] Output: LM head → tokens [batch, seq_len, vocab_size]
```

---

## Architectural Components

### 1. Two-Level Hierarchical Reasoning

**Motivation:** Separate high-level strategic reasoning from low-level detail processing

**Implementation:**
```python
# Two separate latent states
z_H: torch.Tensor  # [batch, seq_len, hidden_size]  - High-level reasoning
z_L: torch.Tensor  # [batch, seq_len, hidden_size]  - Low-level reasoning

# Two-level iteration structure
H_cycles: int = 3     # Outer refinement cycles
L_cycles: int = 4-6   # Inner reasoning cycles
```

**Reasoning Process:**
1. **Low-level (L_level):** Given high-level context + input, process details
   - Input injection: `z_H + input_embeddings`
   - Updates: `z_L` through L_cycles iterations
   - Role: Handle immediate pattern matching and local transformations

2. **High-level (H_level):** Given low-level results, plan strategy
   - Input injection: `z_L` (results from low-level)
   - Updates: `z_H` once per H_cycle
   - Role: Coordinate overall solution strategy

**Key insight:** Same `L_level` module is reused for both high and low-level reasoning through weight sharing!

---

### 2. Recursive Reasoning Block (L_level)

**Purpose:** Core reasoning unit that updates latent states

**Configuration Options:**

**Option A: Transformer-based**
```python
class TRM_Block(nn.Module):
    def __init__(self, config):
        # Self-attention
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            causal=False  # Bidirectional attention
        )

        # Feed-forward network
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion  # e.g., 4x
        )

        self.norm_eps = config.rms_norm_eps

    def forward(self, hidden_states):
        # Post-norm architecture
        # Self-attention + residual + norm
        hidden_states = rms_norm(
            hidden_states + self.self_attn(hidden_states),
            variance_epsilon=self.norm_eps
        )

        # FFN + residual + norm
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps
        )

        return hidden_states
```

**Option B: MLP-transpose (mlp_t)**
```python
# For tasks where sequence mixing is more important than attention
if config.mlp_t:
    self.mlp_t = SwiGLU(
        hidden_size=seq_len + puzzle_emb_len,  # Operates on L dimension
        expansion=config.expansion
    )

    def forward(self, hidden_states):
        # Transpose to operate on sequence dimension
        hidden_states = hidden_states.transpose(1, 2)  # [B, D, L]
        hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states))
        hidden_states = hidden_states.transpose(1, 2)  # [B, L, D]

        # Regular MLP
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states))
        return hidden_states
```

**Component Details:**

- **Attention:** Full bidirectional self-attention (not causal)
- **Activation:** SwiGLU (more expressive than ReLU/GELU)
- **Normalization:** RMS normalization (simpler than LayerNorm)
- **Residual:** Post-norm (norm after addition)
- **Expansion:** 4× hidden dimension in FFN

---

### 3. Reasoning Module Wrapper

```python
class TRM_ReasoningModule(nn.Module):
    def __init__(self, layers: List[TRM_Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # L_layers blocks (e.g., 2)

    def forward(self, hidden_states, input_injection, **kwargs):
        # Inject input (from other level or external input)
        hidden_states = hidden_states + input_injection

        # Pass through all reasoning blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)

        return hidden_states
```

**Key points:**
- Multiple blocks (L_layers = 2 typically)
- Input injection at the start
- Same module reused for both z_H and z_L updates

---

### 4. Input Embeddings

**Multi-source embedding combination:**

```python
def _input_embeddings(self, input_tokens, puzzle_identifiers):
    # 1. Token embeddings (discrete grid values)
    embedding = self.embed_tokens(input_tokens)  # [B, seq_len, hidden]

    # 2. Puzzle embeddings (task-specific context)
    if config.puzzle_emb_ndim > 0:
        puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
        # Reshape and pad to match hidden size
        puzzle_embedding = puzzle_embedding.view(-1, puzzle_emb_len, hidden_size)
        # Prepend to sequence
        embedding = torch.cat((puzzle_embedding, embedding), dim=-2)

    # 3. Position embeddings
    if config.pos_encodings == "rope":
        # Rotary Position Embeddings (RoPE)
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_position_embeddings=seq_len + puzzle_emb_len,
            base=config.rope_theta
        )
    elif config.pos_encodings == "learned":
        # Learned absolute positions
        # Scale by 1/sqrt(2) to maintain variance
        embedding = 0.707106781 * (embedding + self.embed_pos.weight)

    # 4. Scaling (for training stability)
    embed_scale = math.sqrt(hidden_size)
    return embed_scale * embedding
```

**Components:**
- **Token embeddings:** Discrete grid values → continuous vectors
- **Puzzle embeddings:** Task-specific context (initialized to zero)
- **Position encodings:** Either RoPE or learned absolute
- **Embedding scaling:** `sqrt(hidden_size)` for stability

---

### 5. Initial Latent States

```python
# Learnable initial states
self.H_init = nn.Parameter(
    trunc_normal_init_(torch.empty(hidden_size), std=1.0)
)
self.L_init = nn.Parameter(
    trunc_normal_init_(torch.empty(hidden_size), std=1.0)
)

# At start of each sequence (when halted=True)
def reset_carry(self, reset_flag, carry):
    return Carry(
        z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
        z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
    )
```

**Purpose:** Learnable starting points for recursive reasoning

---

### 6. Forward Pass (Recursive Refinement)

**Complete forward iteration:**

```python
def forward(self, carry, batch):
    # Get input embeddings
    input_embeddings = self._input_embeddings(
        batch["inputs"],
        batch["puzzle_identifiers"]
    )

    # Extract latent states from carry
    z_H, z_L = carry.z_H, carry.z_L

    # Recursive refinement with gradient truncation
    # (H_cycles - 1) iterations WITHOUT gradients
    with torch.no_grad():
        for H_step in range(H_cycles - 1):
            # Low-level reasoning (L_cycles iterations)
            for L_step in range(L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings)

            # High-level reasoning (1 iteration)
            z_H = self.L_level(z_H, z_L)

    # Last iteration WITH gradients
    for L_step in range(L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings)
    z_H = self.L_level(z_H, z_L)

    # Generate output
    output = self.lm_head(z_H)[:, puzzle_emb_len:]  # Remove puzzle positions

    # Q-head for adaptive halting
    q_logits = self.q_head(z_H[:, 0])  # Use first position

    # Detach for next carry (no gradient flow across sequences)
    new_carry = Carry(z_H=z_H.detach(), z_L=z_L.detach())

    return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
```

**Key features:**
1. **Gradient truncation:** Only last H_cycle gets gradients (efficiency)
2. **Weight sharing:** Same L_level module used everywhere
3. **Nested loops:** H_cycles (outer) × L_cycles (inner)
4. **Detached carry:** Prevents gradient flow across sequences

---

### 7. Adaptive Computation Time (ACT)

**Purpose:** Learn when to stop refining the answer

**Q-Learning Formulation:**

```python
# Q-head predicts value of halting vs continuing
self.q_head = nn.Linear(hidden_size, 2, bias=True)

# Special initialization for faster learning
with torch.no_grad():
    self.q_head.weight.zero_()
    self.q_head.bias.fill_(-5)  # Start with low Q-values

# Forward
q_logits = self.q_head(z_H[:, 0])  # [batch, 2]
q_halt_logits = q_logits[..., 0]
q_continue_logits = q_logits[..., 1]

# Halting decision (during training)
if training:
    # Halt if Q(halt) > Q(continue)
    halted = (q_halt_logits > q_continue_logits)

    # Force minimum steps (avoid premature halting)
    halted = halted & (steps >= min_halt_steps)

    # Exploration: random minimum halt steps
    if rand() < halt_exploration_prob:
        min_halt_steps = randint(2, halt_max_steps + 1)

    # Compute target Q-value (bootstrapping)
    next_q_halt, next_q_continue = self.forward(carry, batch)
    target_q_continue = sigmoid(max(next_q_halt, next_q_continue))
else:
    # Evaluation: always use max steps (for batching)
    halted = (steps >= halt_max_steps)
```

**Training:**
- **Exploration:** Random minimum halt steps
- **Q-learning:** Bootstrap from next state
- **No replay buffer:** Uses parallel environments (large batch size)
- **No target network:** Similar to PQN approach

**Simplified version (no_ACT_continue=True):**
```python
# Just use sigmoid of halt signal
halted = (q_halt_logits > 0)  # Sigmoid(0) = 0.5 threshold
```

---

### 8. Output Head

```python
# Language modeling head (token prediction)
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

# Generate logits
output = self.lm_head(z_H)  # [batch, seq_len, vocab_size]

# Remove puzzle embedding positions
output = output[:, puzzle_emb_len:]

# Get predictions
predicted_tokens = torch.argmax(output, dim=-1)
```

---

## Training Strategy

### 1. Loss Functions

**Primary: Cross-Entropy Loss**
```python
# Token prediction loss
loss_ce = F.cross_entropy(
    logits.view(-1, vocab_size),
    target_tokens.view(-1),
    ignore_index=IGNORE_LABEL_ID  # -100 for padding
)
```

**Secondary: ACT Q-Learning Losses**
```python
# Q(halt) loss - binary classification
loss_q_halt = F.binary_cross_entropy_with_logits(
    q_halt_logits,
    should_halt.float()
)

# Q(continue) loss - regression to target Q-value
loss_q_continue = F.mse_loss(
    torch.sigmoid(q_continue_logits),
    target_q_continue
)

# Total loss
loss = loss_ce + alpha * loss_q_halt + beta * loss_q_continue
```

### 2. Gradient Truncation

**Efficiency trick:** Only backprop through last H_cycle

```python
# (H_cycles - 1) without gradients
with torch.no_grad():
    for H_step in range(H_cycles - 1):
        for L_step in range(L_cycles):
            z_L = L_level(z_L, z_H + input)
        z_H = L_level(z_H, z_L)

# Last cycle WITH gradients
for L_step in range(L_cycles):
    z_L = L_level(z_L, z_H + input)
z_H = L_level(z_H, z_L)  # Only this gets gradients
```

**Benefits:**
- **Memory efficiency:** Don't store activations from early cycles
- **Training speed:** Less computation in backward pass
- **Still learns:** Final cycle depends on earlier cycles through carry

### 3. Optimizer

```python
# AdamW with specific settings
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1.0,  # Strong regularization
    betas=(0.9, 0.999)
)

# Separate learning rate for puzzle embeddings
puzzle_emb_optimizer = AdamW(
    model.puzzle_emb.parameters(),
    lr=1e-4,  # puzzle_emb_lr
    weight_decay=1.0  # puzzle_emb_weight_decay
)
```

### 4. Training Configuration

**Typical hyperparameters:**
```yaml
# Architecture
hidden_size: 256
num_heads: 8
expansion: 4.0
L_layers: 2
H_cycles: 3
L_cycles: 4

# Training
batch_size: 16  # × 4 GPUs = 64 effective
epochs: 50000
lr: 1e-4
weight_decay: 1.0
eval_interval: 5000

# ACT
halt_max_steps: 3
halt_exploration_prob: 0.1

# EMA (Exponential Moving Average)
ema: True
```

---

## Data Augmentation

**ARC-AGI Puzzles:**
```python
# Each puzzle augmented 1000 times:
- Rotations (90°, 180°, 270°)
- Reflections (horizontal, vertical)
- Color permutations
- Grid transformations

# Dataset size:
Training: ~1000 puzzles × 1000 augments = 1M examples
```

---

## Model Sizes

| Component | Small | Medium | Large |
|-----------|-------|--------|-------|
| **Parameters** | ~3M | ~7M | ~15M |
| **hidden_size** | 128 | 256 | 512 |
| **num_heads** | 4 | 8 | 16 |
| **L_layers** | 2 | 2 | 3 |
| **H_cycles** | 3 | 3 | 4 |
| **L_cycles** | 4 | 4 | 6 |

---

## Key Innovations

### 1. **Weight Sharing**
- Same L_level module for both high and low-level reasoning
- Dramatically reduces parameters while maintaining expressivity
- Enables "recursive reasoning with tiny networks"

### 2. **Two-Level Hierarchy**
- Separates strategic planning (z_H) from detail processing (z_L)
- Mimics cognitive architecture without being tied to neuroscience
- More efficient than flat iteration

### 3. **Gradient Truncation**
- Only backprop through last cycle
- Maintains learning while reducing memory/compute
- Unique training efficiency trick

### 4. **Adaptive Computation**
- Q-learning for learned halting
- Exploration strategy prevents premature stopping
- Bootstrapped Q-values (no replay buffer needed)

### 5. **Post-Norm Architecture**
- Norm after residual addition
- Works well with RMS norm
- Training stability

---

## Differences from Standard Transformers

| Aspect | Standard Transformer | TRM |
|--------|---------------------|-----|
| **Depth** | Many layers (12-96) | Few layers (2-3), many cycles |
| **Weight sharing** | None | Extensive (same module reused) |
| **Computation** | Single forward pass | Recursive refinement (K cycles) |
| **Parameters** | Billions | Millions |
| **Hierarchy** | Flat | Two-level (H/L) |
| **Halting** | Fixed depth | Adaptive (ACT) |
| **Normalization** | LayerNorm (pre/post) | RMS norm (post) |
| **Activation** | GELU/ReLU | SwiGLU |

---

## Performance Results

**ARC-AGI Benchmarks:**
- **ARC-AGI-1:** 45% accuracy (7M parameters)
- **ARC-AGI-2:** 8% accuracy (7M parameters)
- **Sudoku-Extreme:** High accuracy
- **Maze-Hard (30×30):** Strong performance

**Efficiency:**
- **95% fewer parameters** than competitive approaches
- **Training time:** ~3 days on 4× H100 GPUs
- **Memory footprint:** ~20MB model size

---

## Implementation Notes

### Carry State Management

```python
@dataclass
class Carry:
    # Latent states
    z_H: torch.Tensor  # [batch, seq_len, hidden]
    z_L: torch.Tensor  # [batch, seq_len, hidden]

    # ACT state
    steps: torch.Tensor  # [batch] - current step count
    halted: torch.Tensor  # [batch] - bool halted flag

    # Current data
    current_data: Dict[str, torch.Tensor]  # Batch data

# Reset on halt
if halted:
    carry.z_H = H_init  # Reset to learned initial state
    carry.z_L = L_init
    carry.steps = 0
```

### Sequence Batching

**Challenge:** Variable halt times within batch

**Solution:**
- During **training:** Allow variable halting per sequence
- During **evaluation:** Force all sequences to use `halt_max_steps` (uniform batching)

---

## Code Structure

```
TinyRecursiveModels/
├── models/
│   ├── recursive_reasoning/
│   │   ├── trm.py              # Main TRM implementation
│   │   ├── hrm.py              # HRM baseline
│   │   └── transformers_baseline.py
│   ├── layers.py               # SwiGLU, Attention, etc.
│   ├── sparse_embedding.py     # Puzzle embeddings
│   └── common.py
├── dataset/
│   ├── build_arc_dataset.py
│   └── build_sudoku_dataset.py
├── pretrain.py                 # Training script
└── evaluators/                 # Evaluation code
```

---

## Summary

TRM achieves **recursive reasoning with tiny networks** through:

1. **Weight sharing:** Same reasoning module reused across iterations
2. **Two-level hierarchy:** Strategic (z_H) and tactical (z_L) reasoning
3. **Iterative refinement:** K outer × n inner cycles of improvement
4. **Adaptive halting:** Q-learning to decide when to stop
5. **Gradient truncation:** Efficient training by only backpropping last cycle
6. **Parameter efficiency:** 7M params competitive with billion-param models

**Core insight:** Recursive application of small networks can achieve strong reasoning performance without massive scale.
