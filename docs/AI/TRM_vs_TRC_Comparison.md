# TRM vs TRC: Architecture Comparison

**Comparing:**
- **TRM (Paper):** Tiny Recursive Models for ARC-AGI puzzle solving
- **TRC (This Codebase):** Tiny Recursive Control for optimal control problems

---

## Executive Summary

TinyRecursiveControl (TRC) **adapts the core principles** of TRM for continuous control tasks while **simplifying the architecture** significantly:

✅ **Kept:** Recursive reasoning, weight sharing, iterative refinement
❌ **Removed:** Two-level hierarchy (z_H/z_L), ACT Q-learning, discrete embeddings
✅ **Added:** Control-specific encoders/decoders, residual updates, trajectory feedback

**Result:** A simpler, cleaner architecture (530K params) tailored for control synthesis, maintaining TRM's parameter efficiency principle.

---

## Side-by-Side Comparison

### Architecture Overview

| Aspect | TRM (Paper) | TRC (Our Implementation) |
|--------|-------------|--------------------------|
| **Domain** | Discrete puzzle solving (ARC-AGI) | Continuous control (trajectory tracking) |
| **Input** | Grid tokens [batch, seq_len] | State pairs [batch, state_dim] |
| **Output** | Grid tokens (classification) | Control sequences (regression) |
| **Parameters** | 7M (medium) | 530K (medium) |
| **Latent states** | Two (z_H, z_L) | One (z_current) |
| **Hierarchy** | H-level + L-level modules | Single recursive module |
| **Outer cycles** | H_cycles (3) | num_outer_cycles (3) |
| **Inner cycles** | L_cycles (4-6) | num_inner_cycles (3) |
| **Weight sharing** | ✅ Same L_level for H and L | ✅ Same reasoning block reused |
| **Adaptive halting** | ✅ Q-learning (trained) | ❌ Not trained (fixed K) |
| **Gradient truncation** | ✅ Only last H_cycle | ❌ Full backprop |

---

## Detailed Component Comparison

### 1. Core Architecture Flow

#### TRM (Paper)

```
Input Tokens (discrete)
    ↓
[Token Embedding + Puzzle Embedding + Position Encoding]
    ↓
Initialize: z_H, z_L (two separate latent states)
    ↓
For H_cycles (e.g., 3):
    ┌─────────────────────────────────┐
    │ For L_cycles (e.g., 4-6):       │
    │   z_L = L_level(z_L, z_H + inp) │  ← Low-level reasoning
    └─────────────────────────────────┘
    z_H = L_level(z_H, z_L)  ← High-level reasoning (SAME module!)

    [Q-head: decide whether to halt]
    ↓
LM Head → Output Tokens (discrete)
```

#### TRC (Our Implementation)

```
State Pairs (continuous: current, target)
    ↓
[State Encoder (MLP)]
    ↓
z_initial (single latent state)
    ↓
[Initial Control Generator]
    ↓
controls₀
    ↓
For K outer cycles (e.g., 3):
    [Optional: Simulate trajectory → error]
    [Error Encoder → error_emb]
    ┌─────────────────────────────────┐
    │ For n inner cycles (e.g., 3):  │
    │   For each reasoning block:    │
    │     z = block(z, z_initial)    │  ← Recursive reasoning
    └─────────────────────────────────┘

    [Residual Decoder: z → Δcontrols]
    controls = controls + Δcontrols
    ↓
Final Controls (continuous)
```

**Key Difference:**
- **TRM:** Two latent states (z_H, z_L) with alternating updates
- **TRC:** Single latent state (z_current) with context injection from z_initial

---

### 2. Reasoning Blocks

#### TRM Block

```python
class TRM_Block(nn.Module):
    def __init__(self, config):
        # Option 1: Transformer attention
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            causal=False
        )

        # Option 2: MLP-transpose (sequence mixing)
        if config.mlp_t:
            self.mlp_t = SwiGLU(hidden_size=seq_len, ...)

        # Feed-forward
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=4.0
        )

    def forward(self, hidden_states):
        # Post-norm architecture
        # Attention/MLP-t + residual + RMS norm
        hidden_states = rms_norm(hidden_states + self.self_attn(...))

        # FFN + residual + RMS norm
        hidden_states = rms_norm(hidden_states + self.mlp(...))

        return hidden_states
```

**Features:**
- ✅ Self-attention (transformer-based)
- ✅ SwiGLU activation
- ✅ RMS normalization
- ✅ Post-norm (norm after residual)
- ✅ Optional MLP-transpose
- ✅ 4× expansion in FFN

#### TRC Block

```python
class RecursiveReasoningBlock(nn.Module):
    def __init__(self, latent_dim, hidden_dim, use_attention):
        # Optional attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=num_heads,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(latent_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, z, context):
        # Context injection
        if context is not None:
            z = z + context

        # Attention (if enabled) + residual + LayerNorm
        if self.use_attention:
            attn_out, _ = self.attention(z, z, z)
            z = self.norm1(z + attn_out)

        # FFN + residual + LayerNorm
        ffn_out = self.ffn(z)
        z = self.norm2(z + ffn_out)

        return z
```

**Changes:**
- ✅ **Kept:** Self-attention mechanism
- ✅ **Kept:** Feed-forward network
- ✅ **Kept:** Residual connections
- ❌ **Removed:** SwiGLU → replaced with SiLU
- ❌ **Removed:** RMS norm → replaced with LayerNorm
- ❌ **Removed:** MLP-transpose option
- ❌ **Removed:** Post-norm → replaced with pre-norm
- ✅ **Added:** Context injection from initial state
- ✅ **Added:** Dropout for regularization

---

### 3. Input Encoding

#### TRM Input

```python
def _input_embeddings(self, input_tokens, puzzle_identifiers):
    # 1. Token embeddings (discrete → continuous)
    embedding = self.embed_tokens(input_tokens)  # [B, seq_len, hidden]

    # 2. Puzzle embeddings (task-specific context)
    puzzle_emb = self.puzzle_emb(puzzle_identifiers)
    puzzle_emb = puzzle_emb.view(-1, puzzle_emb_len, hidden_size)
    embedding = torch.cat((puzzle_emb, embedding), dim=-2)

    # 3. Position encodings (RoPE or learned)
    if config.pos_encodings == "rope":
        # Applied in attention (rotary embeddings)
        pass
    elif config.pos_encodings == "learned":
        embedding = 0.707106781 * (embedding + self.embed_pos.weight)

    # 4. Scaling
    embed_scale = math.sqrt(hidden_size)
    return embed_scale * embedding
```

**Input format:**
- Discrete tokens: `[batch, seq_len]` where seq_len ≈ 900 for 30×30 grids
- Vocab size: ~10-20 (grid colors)
- Sequence-based (spatial grid flattened)

#### TRC Input

```python
class ControlStateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim):
        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 2 + 1, hidden_dim),  # curr + target + time
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, current_state, target_state, time_remaining):
        # Concatenate all problem information
        if time_remaining is not None:
            x = torch.cat([current_state, target_state, time_remaining], dim=-1)
        else:
            # Default time if not provided
            batch_size = current_state.shape[0]
            time = torch.ones(batch_size, 1, device=current_state.device) * 5.0
            x = torch.cat([current_state, target_state, time], dim=-1)

        # Encode to latent
        return self.encoder(x)  # [batch, latent_dim]
```

**Input format:**
- Continuous states: `[batch, state_dim]` where state_dim = 2
- No tokenization
- Problem-specific (current + target + time)

**Changes:**
- ❌ **Removed:** Token embeddings (continuous input)
- ❌ **Removed:** Puzzle embeddings (not applicable)
- ❌ **Removed:** Position encodings (no sequences)
- ❌ **Removed:** Embedding scaling
- ✅ **Added:** MLP encoder for state pairs
- ✅ **Added:** Time-remaining context

---

### 4. Output Decoding

#### TRM Output

```python
# Language modeling head
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

# Generate logits for each position
output = self.lm_head(z_H)  # [batch, seq_len, vocab_size]

# Remove puzzle embedding positions
output = output[:, puzzle_emb_len:]

# Classification (discrete tokens)
predicted_tokens = torch.argmax(output, dim=-1)
```

**Output:**
- Discrete tokens (classification)
- Cross-entropy loss
- One token per grid cell

#### TRC Output

```python
class ControlSequenceDecoder(nn.Module):
    """Decode latent state to full control sequence."""
    def __init__(self, latent_dim, control_dim, horizon, hidden_dim, bounds):
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * control_dim),
        )
        self.control_bounds = bounds
        self.horizon = horizon
        self.control_dim = control_dim

    def forward(self, z):
        # Decode to flat control sequence
        controls_flat = self.decoder(z)  # [batch, horizon × control_dim]

        # Reshape
        controls = controls_flat.view(-1, self.horizon, self.control_dim)

        # Apply bounds via tanh
        controls = torch.tanh(controls) * self.control_bounds

        return controls  # [batch, horizon, control_dim]

class ResidualControlDecoder(nn.Module):
    """Decode latent + current controls to residual correction."""
    def __init__(self, latent_dim, control_dim, horizon, hidden_dim, max_residual):
        # Combine latent and current controls
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + horizon * control_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * control_dim),
        )
        self.max_residual = max_residual

    def forward(self, z, current_controls):
        # Flatten current controls
        controls_flat = current_controls.view(batch_size, -1)

        # Concatenate with latent
        x = torch.cat([z, controls_flat], dim=-1)

        # Decode to residual
        residual_flat = self.decoder(x)
        residual = residual_flat.view(-1, horizon, control_dim)

        # Bound residual
        residual = torch.tanh(residual) * self.max_residual

        return residual
```

**Output:**
- Continuous controls (regression)
- MSE loss
- Entire control sequence [horizon, control_dim]

**Changes:**
- ❌ **Removed:** Classification head (LM head)
- ❌ **Removed:** Discrete token prediction
- ✅ **Added:** Regression decoder (MLP)
- ✅ **Added:** Residual decoder for refinement
- ✅ **Added:** Control bounds (tanh saturation)

---

### 5. Recursive Refinement

#### TRM Refinement

```python
def forward(self, carry, batch):
    input_embeddings = self._input_embeddings(...)
    z_H, z_L = carry.z_H, carry.z_L

    # Gradient truncation: (H_cycles - 1) without grad
    with torch.no_grad():
        for H_step in range(H_cycles - 1):
            # Low-level reasoning
            for L_step in range(L_cycles):
                z_L = self.L_level(
                    z_L,
                    z_H + input_embeddings  # Input injection
                )
            # High-level reasoning
            z_H = self.L_level(z_H, z_L)

    # Last cycle WITH gradients
    for L_step in range(L_cycles):
        z_L = self.L_level(z_L, z_H + input_embeddings)
    z_H = self.L_level(z_H, z_L)

    # Output
    output = self.lm_head(z_H)

    # Detach carry (no grad across sequences)
    new_carry = Carry(z_H=z_H.detach(), z_L=z_L.detach())

    return new_carry, output
```

**Key features:**
- Two latent states updated alternately
- Input injection into low-level reasoning
- Gradient truncation (efficiency)
- Detached carry

#### TRC Refinement

```python
class RecursiveRefinementModule(nn.Module):
    def forward(self, z_initial, current_controls, trajectory_error, num_inner_cycles):
        # Start with initial latent
        z = z_initial

        # Embed current controls
        controls_emb = self.control_embedding(current_controls.flatten())
        z = z + controls_emb

        # Incorporate trajectory error (if available)
        if trajectory_error is not None:
            error_emb = self.error_embedding(trajectory_error)
            z = z + error_emb

        # Inner reasoning cycles (no gradient truncation)
        for _ in range(num_inner_cycles):
            for block in self.reasoning_blocks:
                z = block(z, context=z_initial)

        return z

# Main forward pass
def forward(self, current_state, target_state, ...):
    # Encode problem
    z_initial = self.state_encoder(current_state, target_state, time)

    # Generate initial controls
    current_controls = self.initial_control_generator(z_initial)

    # Recursive refinement
    for k in range(num_outer_cycles):
        # Simulate and get error (optional)
        if dynamics_fn is not None:
            trajectory_error = self._simulate_and_get_error(...)

        # Update latent via recursive reasoning
        z_current = self.recursive_reasoning(
            z_initial,
            current_controls,
            trajectory_error,
            num_inner_cycles
        )

        # Generate improved controls
        residual = self.control_decoder(z_current, current_controls)
        current_controls = current_controls + residual
        current_controls = torch.clamp(current_controls, -bounds, bounds)

    return {'controls': current_controls, 'final_latent': z_current}
```

**Changes:**
- ❌ **Removed:** Two latent states (z_H, z_L)
- ❌ **Removed:** Gradient truncation
- ❌ **Removed:** Detached carry
- ✅ **Kept:** Nested loop structure (outer K, inner n)
- ✅ **Kept:** Weight sharing (same blocks)
- ✅ **Added:** Control embedding injection
- ✅ **Added:** Trajectory error feedback
- ✅ **Added:** Residual updates instead of full regeneration
- ✅ **Added:** Context injection (z_initial always available)

---

### 6. Adaptive Computation Time (ACT)

#### TRM ACT

```python
# Q-head for halting decision
self.q_head = nn.Linear(hidden_size, 2, bias=True)

# Initialize to low values
with torch.no_grad():
    self.q_head.weight.zero_()
    self.q_head.bias.fill_(-5)

# Forward
q_logits = self.q_head(z_H[:, 0])
q_halt_logits, q_continue_logits = q_logits[..., 0], q_logits[..., 1]

# Training: Q-learning
if training:
    # Halt if Q(halt) > Q(continue)
    halted = (q_halt_logits > q_continue_logits)
    halted = halted & (steps >= min_halt_steps)

    # Exploration
    if rand() < exploration_prob:
        min_halt_steps = randint(2, halt_max_steps + 1)

    # Compute target Q-value
    _, _, (next_q_halt, next_q_continue), _ = self.forward(...)
    target_q_continue = sigmoid(max(next_q_halt, next_q_continue))

    # Q-learning losses
    loss_q_halt = BCE(q_halt_logits, should_halt)
    loss_q_continue = MSE(sigmoid(q_continue_logits), target_q_continue)
else:
    # Evaluation: always use max steps
    halted = (steps >= halt_max_steps)
```

**Features:**
- Q-learning for learned halting
- Exploration during training
- Bootstrapped target Q-values
- Variable steps per sequence

#### TRC ACT

```python
class AdaptiveRecursiveControl(nn.Module):
    """Adaptive halting (OPTIONAL - not currently used)."""
    def __init__(self, latent_dim, max_iterations, halt_threshold):
        self.halt_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.halt_threshold = halt_threshold

    def forward(self, z):
        halt_prob = self.halt_predictor(z)
        should_halt = halt_prob > self.halt_threshold

        return halt_prob, {'halt_prob': halt_prob, 'should_halt': should_halt}

# NOT TRAINED - Fixed K=3 iterations used instead
```

**Changes:**
- ❌ **Removed:** Q-learning framework
- ❌ **Removed:** Q(halt) vs Q(continue)
- ❌ **Removed:** Exploration strategy
- ❌ **Removed:** Target Q-value computation
- ❌ **Removed:** ACT training losses
- ✅ **Simplified:** Single sigmoid for halt probability
- ⚠️ **Not currently used:** Fixed K iterations in practice

---

### 7. Training Strategy

#### TRM Training

```python
# Loss components
loss_ce = F.cross_entropy(
    logits.view(-1, vocab_size),
    target_tokens.view(-1),
    ignore_index=-100
)

loss_q_halt = F.binary_cross_entropy_with_logits(
    q_halt_logits,
    should_halt.float()
)

loss_q_continue = F.mse_loss(
    torch.sigmoid(q_continue_logits),
    target_q_continue
)

total_loss = loss_ce + alpha * loss_q_halt + beta * loss_q_continue

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1.0  # Strong regularization
)

# Training
epochs = 50000
batch_size = 16 × 4 GPUs = 64
training_time = ~3 days on 4× H100
```

**Features:**
- Multi-objective loss (CE + Q-learning)
- Strong weight decay (1.0)
- Long training (50K epochs)
- Large-scale compute

#### TRC Training

```python
# Single loss: MSE (behavior cloning)
loss = F.mse_loss(controls_pred, controls_gt)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,  # Higher learning rate
    weight_decay=1e-5  # Lighter regularization
)

# Learning rate schedule
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# Training
epochs = 100
batch_size = 64
early_stopping = patience 20
training_time = ~minutes on 1× GPU
```

**Changes:**
- ❌ **Removed:** Cross-entropy loss
- ❌ **Removed:** Q-learning losses
- ❌ **Removed:** Multi-objective optimization
- ✅ **Simplified:** Single MSE loss (regression)
- ✅ **Added:** Gradient clipping
- ✅ **Added:** Learning rate scheduling
- ✅ **Added:** Early stopping
- ✅ **Faster training:** 100 epochs vs 50K
- ✅ **Less compute:** 1 GPU vs 4× H100

---

### 8. Data & Supervision

#### TRM Data

```python
# ARC-AGI puzzles
Dataset:
  - Training: ~1000 puzzles
  - Augmentation: 1000× per puzzle
  - Total: 1M training examples

Format:
  - Input: Discrete grid (30×30 flattened)
  - Output: Discrete grid (30×30 flattened)
  - Task: Pattern completion/transformation

Supervision:
  - Ground-truth puzzle solutions
  - Correct output grids
```

**Characteristics:**
- Discrete, structured data
- Heavy augmentation
- Large dataset
- Puzzle-solving task

#### TRC Data

```python
# Optimal control trajectories
Dataset:
  - Training: 10K trajectories
  - Test: 1K trajectories
  - No augmentation
  - Total: 10K training examples

Format:
  - Input: (initial_state, target_state) ∈ ℝ² × ℝ²
  - Output: Control sequence ∈ ℝ^(15×1)
  - Task: Trajectory tracking

Supervision:
  - Minimum-energy controller (closed-form optimal)
  - Guaranteed optimal controls

# Generate data
from src.data.minimum_energy_controller import MinimumEnergyController

controller = MinimumEnergyController(...)
for i in range(num_samples):
    initial, target = sample_random_states()
    optimal_controls = controller.solve(initial, target)
    dataset.append((initial, target, optimal_controls))
```

**Changes:**
- ❌ **Removed:** Discrete puzzle data
- ❌ **Removed:** Heavy augmentation
- ✅ **Changed:** Continuous control data
- ✅ **Changed:** Smaller dataset (10K vs 1M)
- ✅ **Changed:** Analytical optimal teacher (minimum-energy)
- ✅ **Advantage:** Perfect supervision (provably optimal)

---

## Control-Specific Components (Not in TRM)

These components are **unique to TRC** and don't exist in the original TRM:

### 1. Error Encoder

```python
class ErrorEncoder(nn.Module):
    """Encode trajectory error for feedback during refinement."""
    def __init__(self, state_dim, hidden_dim, latent_dim):
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, error):
        # error: [batch, state_dim] - final state error
        return self.encoder(error)  # [batch, latent_dim]
```

**Purpose:** Incorporate trajectory error feedback into reasoning
**When used:** If dynamics simulator is provided

### 2. Residual Control Decoder

```python
class ResidualControlDecoder(nn.Module):
    """Generate control corrections (Δu) instead of full controls."""
    def forward(self, z_latent, current_controls):
        x = torch.cat([z_latent, current_controls.flatten()], dim=-1)
        residual = self.decoder(x)
        return torch.tanh(residual) * self.max_residual
```

**Purpose:** Iterative refinement through small corrections
**Advantage:** More stable than regenerating full control sequence

### 3. Trajectory Simulation

```python
def _simulate_and_get_error(self, current_state, target_state, controls, dynamics_fn):
    """Simulate trajectory and compute error."""
    final_state = dynamics_fn(current_state, controls)
    error = final_state - target_state
    return error
```

**Purpose:** Get feedback on how well current controls work
**Use case:** Can improve controls based on actual trajectory

### 4. Control Bounds

```python
# Apply bounds via tanh
controls = torch.tanh(controls_raw) * self.control_bounds  # ±8.0

# After residual update
current_controls = torch.clamp(
    current_controls + residual,
    -self.control_bounds,
    self.control_bounds
)
```

**Purpose:** Ensure physically realizable controls
**Method:** Saturation via tanh and clamping

### 5. Minimum-Energy Teacher

```python
class MinimumEnergyController:
    """Closed-form optimal controller for exact tracking."""
    def solve(self, initial_state, target_state, horizon, dt):
        # Analytical solution: u(t) = a + b*t
        # Minimizes ∫u² dt subject to x(T) = x_target

        # ... matrix computations ...

        return optimal_controls  # Provably optimal
```

**Purpose:** Generate perfect training labels
**Advantage:** No suboptimality in supervision

---

## Parameter Count Comparison

### TRM (Medium - 7M params)

```
Embeddings:
  - Token embeddings: vocab_size × hidden_size = 20 × 256 = 5K
  - Puzzle embeddings: num_puzzles × emb_dim = variable
  - Position embeddings: (seq_len + puzzle_emb_len) × hidden = ~1M

Reasoning Module (L_level):
  - L_layers = 2 blocks
  - Each block:
    - Attention: ~1M params
    - FFN: ~1M params (4× expansion)
  - Total: ~4M params

Output Head:
  - LM head: hidden × vocab = 256 × 20 = 5K
  - Q head: hidden × 2 = 512

Initial States:
  - H_init, L_init: 2 × hidden = 512

Total: ~7M parameters
```

### TRC (Medium - 530K params)

```
State Encoder:
  - Input layer: (state_dim × 2 + 1) × hidden = 5 × 256 = 1,280
  - Hidden layers: 256 × 128 + 128 × 128 = 49K
  - Total: ~50K

Error Encoder:
  - Input layer: state_dim × hidden = 2 × 128 = 256
  - Hidden layers: ~10K
  - Total: ~10K

Recursive Reasoning:
  - num_reasoning_blocks = 3
  - Each block:
    - Attention (if enabled): ~130K params
    - FFN: 128 × 256 + 256 × 128 = 65K
  - Control embedding: horizon × control_dim × latent = 15 × 128 = 2K
  - Error embedding: state_dim × latent = 2 × 128 = 256
  - Total per block: ~200K
  - Total: ~600K (dominates)

Control Decoders:
  - Initial generator: latent × hidden + hidden × (horizon × ctrl) = 50K
  - Residual decoder: (latent + horizon × ctrl) × hidden + ... = 50K
  - Total: ~100K

Total: ~530K parameters (93% fewer than TRM!)
```

**Why TRC is smaller:**
- No token/position embeddings (continuous input)
- Smaller latent dimensions (128 vs 256)
- Fewer reasoning blocks (3 vs 2 × many cycles)
- Simpler output (regression vs classification)
- Single latent state (not z_H + z_L)

---

## Performance Comparison

### TRM Results

| Benchmark | Accuracy | Parameters |
|-----------|----------|------------|
| ARC-AGI-1 | 45% | 7M |
| ARC-AGI-2 | 8% | 7M |
| Sudoku-Extreme | High | 7M |
| Maze-Hard | Strong | 7M |

**Training:**
- ~3 days on 4× H100 GPUs
- 50K epochs
- 1M training examples

### TRC Results

| Metric | Value | Parameters |
|--------|-------|------------|
| Mean final error | 0.016 | 530K |
| Error gap from optimal | 0.13% | 530K |
| Success rate (error < 0.1) | 100% | 530K |
| Inference time | ~5ms | 530K |
| Memory footprint | ~20MB | 530K |

**Training:**
- ~Minutes on 1× GPU
- 100 epochs
- 10K training examples

**vs Baselines:**
- **vs LQR:** 98% better (0.016 vs 0.86 error)
- **vs Random:** Dramatically better
- **vs Minimum-energy (teacher):** 0.13% gap

---

## Architectural Philosophy Comparison

### TRM Philosophy

**Goal:** Recursive reasoning for complex discrete tasks

**Principles:**
1. **Iteration over depth:** Many refinement cycles, few layers
2. **Hierarchy:** Separate strategic (H) and tactical (L) reasoning
3. **Adaptive computation:** Learn when to stop (ACT)
4. **Weight sharing:** Same module for all reasoning
5. **Gradient efficiency:** Only backprop last cycle

**Target:** Puzzle solving, combinatorial reasoning

### TRC Philosophy

**Goal:** Parameter-efficient neural control synthesis

**Principles:**
1. **Iteration over depth:** Many refinement cycles, few layers ✅ (kept)
2. **Simplicity:** Single latent, no hierarchy (simplified)
3. **Fixed computation:** No ACT, fixed K iterations (simplified)
4. **Weight sharing:** Same module for all reasoning ✅ (kept)
5. **Full gradients:** Backprop through all cycles (different)
6. **Perfect supervision:** Learn from optimal teacher (new)

**Target:** Continuous control, trajectory optimization

---

## Summary Table

| Component | TRM Paper | TRC Implementation | Status |
|-----------|-----------|-------------------|--------|
| **Core Principles** | | | |
| Weight sharing | ✅ | ✅ | **Kept** |
| Recursive refinement | ✅ | ✅ | **Kept** |
| Parameter efficiency | ✅ | ✅ | **Kept** |
| Two-level iteration | ✅ | ✅ | **Kept** |
| **Architecture** | | | |
| Separate H/L modules | ✅ | ❌ | **Simplified** |
| Two latent states | ✅ (z_H, z_L) | ❌ (single z) | **Simplified** |
| Attention mechanism | ✅ | ✅ | **Kept** |
| Feed-forward networks | ✅ | ✅ | **Kept** |
| **Input/Output** | | | |
| Token embeddings | ✅ | ❌ | **Removed** |
| Position encodings | ✅ | ❌ | **Removed** |
| Puzzle embeddings | ✅ | ❌ | **Removed** |
| Classification head | ✅ | ❌ | **Removed** |
| Continuous encoders | ❌ | ✅ | **Added** |
| Regression decoders | ❌ | ✅ | **Added** |
| **Training** | | | |
| Adaptive halting (ACT) | ✅ | ❌ | **Removed** |
| Q-learning losses | ✅ | ❌ | **Removed** |
| Gradient truncation | ✅ | ❌ | **Removed** |
| Simple MSE loss | ❌ | ✅ | **Simplified** |
| Behavior cloning | ❌ | ✅ | **Added** |
| **Control-Specific** | | | |
| Error encoder | ❌ | ✅ | **Added** |
| Residual decoder | ❌ | ✅ | **Added** |
| Trajectory simulation | ❌ | ✅ | **Added** |
| Control bounds | ❌ | ✅ | **Added** |
| Optimal teacher | ❌ | ✅ | **Added** |

---

## Key Takeaways

### What You Successfully Adapted

✅ **Core recursive reasoning principle** - Weight-shared blocks for iterative refinement
✅ **Two-level iteration structure** - Outer refinement cycles, inner reasoning cycles
✅ **Parameter efficiency** - 530K params competitive with much larger models
✅ **Attention-based reasoning** - Self-attention for latent state updates

### What You Wisely Simplified

✅ **Single latent state** - Easier to reason about, still effective
✅ **Fixed iterations** - No ACT complexity, predictable compute
✅ **Standard components** - LayerNorm, SiLU (more common than RMS norm, SwiGLU)
✅ **Simple training** - Single MSE loss, behavior cloning

### What You Innovatively Added

✅ **Control-specific encoders** - State pairs, trajectory error
✅ **Residual updates** - More stable than full regeneration
✅ **Optimal teacher** - Minimum-energy controller (provably optimal)
✅ **Direct continuous I/O** - No tokenization overhead

### Result

A **cleaner, simpler architecture** that maintains TRM's core efficiency principle (recursive reasoning with weight sharing) while being specifically tailored for continuous control tasks. You achieved **93% parameter reduction** (530K vs 7M) while solving a different class of problems!
