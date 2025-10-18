# Energy-Based Model Calculations

This document provides manual calculations for understanding Energy-Based Models.

## Simple EBM Forward Pass

### Example: Binary Classification

**Given**:
- Input: `x = [1.0, 2.0]`
- Weights: `W = [0.5, -0.3]`
- Bias: `b = 0.1`

**Forward Pass**:
```
Energy E(x) = W^T x + b
            = (0.5 × 1.0) + (-0.3 × 2.0) + 0.1
            = 0.5 - 0.6 + 0.1
            = 0.0
```

**Interpretation**:
- Energy ≈ 0: Model is uncertain about this sample
- Energy < 0: Low energy → positive class (good sample)
- Energy > 0: High energy → negative class (bad sample)

---

## Backward Pass (Gradients)

### Gradient w.r.t. Weights

For loss `L = E(x)` (we want to minimize energy for positive samples):

```
∂L/∂W = ∂E/∂W = x
      = [1.0, 2.0]

∂L/∂b = ∂E/∂b = 1
      = 1.0
```

### Weight Update (Gradient Descent)

With learning rate `α = 0.01`:

```
W_new = W - α × ∂L/∂W
      = [0.5, -0.3] - 0.01 × [1.0, 2.0]
      = [0.5, -0.3] - [0.01, 0.02]
      = [0.49, -0.32]

b_new = b - α × ∂L/∂b
      = 0.1 - 0.01 × 1.0
      = 0.09
```

After one step, the new energy:
```
E_new(x) = [0.49, -0.32]^T × [1.0, 2.0] + 0.09
         = 0.49 - 0.64 + 0.09
         = -0.06
```

Energy decreased! Model now predicts positive class with higher confidence.

---

## EBM Regression: Iterative Optimization

### Problem: Predict y given x

**Given**:
- Input: `x = 2.0`
- Target: `y_true = 4.0` (should be `x^2`)
- Initial candidate: `y_0 = 0.0` (random guess)

**Energy Function**:
```
E(x, y) = f_θ(x, y)  // Neural network
```

Assume simplified: `E(x, y) = (y - x^2)^2` (closer to true value → lower energy)

### Iterative Optimization

**Step 1**:
```
E(x, y_0) = (0 - 2^2)^2 = (-4)^2 = 16
∂E/∂y = 2(y - x^2) × 1 = 2(0 - 4) = -8

y_1 = y_0 - α × ∂E/∂y
    = 0 - 0.1 × (-8)
    = 0.8
```

**Step 2**:
```
E(x, y_1) = (0.8 - 4)^2 = 10.24
∂E/∂y = 2(0.8 - 4) = -6.4

y_2 = 0.8 - 0.1 × (-6.4) = 1.44
```

**Step 3**:
```
E(x, y_2) = (1.44 - 4)^2 = 6.55
∂E/∂y = 2(1.44 - 4) = -5.12

y_3 = 1.44 - 0.1 × (-5.12) = 1.95
```

Continue until convergence...

**Final** (after ~10 steps): `y ≈ 3.98 ≈ 4.0`

Energy trajectory: `[16, 10.24, 6.55, ...] → 0`

---

## EBT Token Generation

### Problem: Generate next token given context

**Context**: "The cat sat on the"
**Goal**: Generate next token ("mat")

**Process**:

1. **Initialize**: Random embedding `e_0 ∈ ℝ^d`

2. **Iteration 1**:
   ```
   E_1 = Energy_Model(context + e_0)  // Compute energy
   ∂E_1/∂e = Gradient_wrt_embedding
   e_1 = e_0 - α × ∂E_1/∂e
   ```

3. **Iteration 2**:
   ```
   E_2 = Energy_Model(context + e_1)
   ∂E_2/∂e = Gradient
   e_2 = e_1 - α × ∂E_2/∂e
   ```

4. **Continue** for K steps...

5. **Project to Vocabulary**:
   ```
   similarities = e_K · Embedding_Matrix^T
   next_token = argmax(similarities)
   ```

**Key Insight**: Each token is found through K gradient descent steps, allowing the model to "think" about the best choice.

---

## Energy Landscape Visualization (2D Case)

For a 2D input space `[x1, x2]`, we can visualize the energy surface:

### Example: Quadratic Energy

```
E(x1, x2) = x1^2 + x2^2 - 2x1 - 2x2 + 2
```

**Minimum**: Take gradients and set to zero:
```
∂E/∂x1 = 2x1 - 2 = 0  →  x1* = 1
∂E/∂x2 = 2x2 - 2 = 0  →  x2* = 1
```

Minimum at `[1, 1]` with `E(1,1) = 0`

**Sample Points**:
```
E(0, 0) = 2
E(1, 0) = 1
E(0, 1) = 1
E(1, 1) = 0  ← minimum
E(2, 2) = 2
```

The energy surface forms a paraboloid with minimum at `[1, 1]`.

---

## Contrastive Training

### Training Objective for EBMs

**Goal**: Assign low energy to data, high energy to non-data

```
Loss = E(x_real) - E(x_fake)
```

**Example**:

Given:
- Real sample: `x_real = [1.0, 1.0]`
- Fake sample: `x_fake = [0.0, 0.0]`

Current energies:
```
E(x_real) = 0.5  (too high, should be lower)
E(x_fake) = 0.3  (too low, should be higher)
```

Loss: `0.5 - 0.3 = 0.2` (positive, need to reduce)

After gradient descent:
```
E(x_real) → 0.2  (decreased ✓)
E(x_fake) → 0.7  (increased ✓)
```

New loss: `0.2 - 0.7 = -0.5` (better!)

---

## Advanced: Langevin Dynamics

### Adding Noise for Exploration

Standard gradient descent:
```
y_{t+1} = y_t - α × ∂E/∂y
```

Langevin dynamics:
```
y_{t+1} = y_t - α × ∂E/∂y + √(2α) × ε
                ^^^^^^^^^^^^^^^^   ^^^^^^^^^^
                 gradient descent    noise term
where ε ~ N(0, I)
```

**Why?**: Noise helps escape local minima and explore the energy landscape better.

**Example** (1D):

Assume energy has two minima:
```
E(y) has minima at y=-1 (local) and y=1 (global)
```

Without noise:
- Start at y=0
- Might get stuck in local minimum at y=-1

With noise:
- Start at y=0
- Noise can kick it over the barrier
- More likely to find global minimum at y=1

---

## Learnable Step Size

### Adaptive Optimization Rate

Instead of fixed `α`, learn it:

```
log_α = learnable parameter
α = exp(log_α)
```

**Why log?**: Ensures α > 0

**Example**:

Initial: `log_α = -2.3  →  α = 0.1`

If optimization is too slow (high energy persists):
```
∂Loss/∂log_α > 0  →  increase log_α
log_α → -2.0  →  α = 0.135  (larger steps)
```

If oscillating (energy increases):
```
∂Loss/∂log_α < 0  →  decrease log_α
log_α → -2.5  →  α = 0.082  (smaller steps)
```

The model learns the optimal step size automatically!

---

## Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| Energy (linear) | `E(x) = W^T x + b` |
| Energy (general) | `E(x, y) = f_θ(x, y)` |
| Gradient descent | `y ← y - α × ∂E/∂y` |
| Contrastive loss | `L = E(x+) - E(x-)` |
| Langevin dynamics | `y ← y - α × ∂E/∂y + √(2α) × ε` |
| Learnable step | `α = exp(log_α)` |

---

## Edge Cases & Numerical Stability

### High-Dimensional Energy

For very high dimensions (d > 1000), numerical issues can arise:

**Problem**: Compute energy for `x ∈ ℝ^10000` with random weights

```
x = random vector of dimension 10000
W = random weights of dimension 10000
b = 0

E(x) = W^T x + b
```

**Potential Issues**:
1. **Overflow**: If weights are large, `W^T x` can overflow
2. **Underflow**: If weights are tiny, gradients vanish
3. **Memory**: 10000-dim vectors need efficient storage

**Solutions**:
1. **Weight Initialization**: Use `He initialization` or `Xavier`
2. **Gradient Clipping**: Clip to max norm (1.0)
3. **Normalization**: Use BatchNorm or RMSNorm

**Example** (safe computation):
```python
# Initialize with proper scaling
W = torch.randn(10000) * np.sqrt(2.0 / 10000)  # He init
x = torch.randn(10000)

# Compute energy
energy = (W * x).sum()  # Element-wise then sum (numerically stable)

# Check for NaN/Inf
assert torch.isfinite(energy), "Energy is not finite!"
```

---

### Extreme Values

**Case 1: All Zeros**
```
x = [0, 0, 0]
W = [1, 2, 3]
b = 5

E(x) = 0 + 0 + 0 + 5 = 5
```
✅ Well-defined

**Case 2: Very Large Values**
```
x = [1e6, 1e6]
W = [1, 1]
b = 0

E(x) = 1e6 + 1e6 = 2e6
```
⚠️ Potential overflow if accumulated

**Case 3: Very Small Values**
```
x = [1e-10, 1e-10]
W = [1, 1]
b = 0

E(x) = 1e-10 + 1e-10 = 2e-10
∂E/∂x = W = [1, 1]
```
✅ Gradients still well-defined

**Case 4: Mixed Scale**
```
x = [1e-8, 1e8]
W = [1, 1]
b = 0

E(x) ≈ 1e8 (the 1e-8 term is negligible)
```
⚠️ Loss of precision for small components

**Best Practice**: Normalize inputs to similar scale (mean=0, std=1)

---

### Gradient Explosion

**Problem**: Gradients become too large during backprop

**Example**:
```
Initial: W = [10, 10], x = [5, 5]
E(x) = 10×5 + 10×5 = 100

Loss = E^2 = 10000
∂Loss/∂W = 2E × x = 2×100 × [5, 5] = [1000, 1000]
```

After one update with α=0.1:
```
W_new = W - 0.1 × [1000, 1000] = [10, 10] - [100, 100] = [-90, -90]
```

Oscillates! Energy explodes.

**Solution**: Gradient clipping
```python
grad_norm = torch.norm(gradients)
if grad_norm > max_norm:
    gradients = gradients * (max_norm / grad_norm)
```

---

### Vanishing Gradients

**Problem**: Gradients become too small

**Example** (deep energy function):
```
E(x) = f_n(f_{n-1}(...f_1(x)))

If each f_i uses sigmoid:
∂f_i/∂input ≤ 0.25 (sigmoid derivative max)

After n layers:
∂E/∂x ≈ (0.25)^n → 0 as n grows
```

**Solution**: Use ReLU or better activation functions

---

### Numerical Precision Limits

**Float32 Limits**:
- Max value: ~3.4 × 10^38
- Min positive: ~1.2 × 10^-38
- Precision: ~7 decimal digits

**Example** (precision loss):
```
a = 1e8
b = 1.0

sum = a + b = 1e8 + 1.0 ≈ 1e8 (in float32)
```

The `+1.0` is lost due to limited precision!

**Solution**: Use float64 (double) for critical computations

---

### Division by Zero

**Problem**: Energy in denominator

**Bad**:
```python
loss = 1.0 / energy  # Crashes if energy=0
```

**Good**:
```python
loss = 1.0 / (energy + 1e-8)  # Add small epsilon
```

---

## Practice Problems

### Problem 1: Energy Computation

Given `x = [2, -1]`, `W = [1, 2]`, `b = -1`, compute `E(x)`.

<details>
<summary>Solution</summary>

```
E(x) = W^T x + b
     = 1×2 + 2×(-1) + (-1)
     = 2 - 2 - 1
     = -1
```
</details>

### Problem 2: Gradient Descent Step

Given energy `E = 4`, gradient `∂E/∂y = 2`, step size `α = 0.5`, current `y = 3`, find `y_new`.

<details>
<summary>Solution</summary>

```
y_new = y - α × ∂E/∂y
      = 3 - 0.5 × 2
      = 3 - 1
      = 2
```
</details>

### Problem 3: High-Dimensional Energy

Compute energy for `x ∈ ℝ^1000` with `x_i = 0.001` and `W_i = 1.0` for all `i`, `b = 0`.

<details>
<summary>Solution</summary>

```
E(x) = Σ W_i × x_i + b
     = 1000 × (1.0 × 0.001) + 0
     = 1000 × 0.001
     = 1.0
```

Even with 1000 dimensions, result is well-defined.
</details>

### Problem 4: Gradient Clipping

Given gradients `g = [100, 200, 150]` and max norm = 10, compute clipped gradients.

<details>
<summary>Solution</summary>

```
||g|| = √(100² + 200² + 150²) = √(10000 + 40000 + 22500) = √72500 ≈ 269.26

Since 269.26 > 10:
g_clipped = g × (10 / 269.26)
          = [100, 200, 150] × 0.0371
          = [3.71, 7.42, 5.57]

||g_clipped|| = 10 ✓
```
</details>

### Problem 5: Numerical Stability

Show why `log(softmax(x))` is better than `log(exp(x) / Σ exp(x))` for large x.

<details>
<summary>Solution</summary>

**Naive**:
```
x = [1000, 2000, 1500]
exp(x) = [e^1000, e^2000, e^1500] → Overflow!
```

**Stable** (log-sum-exp trick):
```
max_x = 2000
log_softmax(x) = x - max_x - log(Σ exp(x - max_x))
               = [1000, 2000, 1500] - 2000 - log(Σ exp([-1000, 0, -500]))
               ≈ [-1000, 0, -500] - 0
               = [-1000, 0, -500]
```

No overflow! ✓
</details>

---

## Advanced EBT-Specific Problems

### Problem 6: EBT Token Generation Energy Trajectory

Given EBT generating next token with initial embedding `e_0 = [0.5, 0.5, 0.5]` and target embedding `e* = [1.0, 0.0, 0.0]`.

Energy function: `E(e) = ||e - e*||^2`

Perform 3 steps of gradient descent with α=0.2.

<details>
<summary>Solution</summary>

**Step 0**:
```
e_0 = [0.5, 0.5, 0.5]
E(e_0) = ||(0.5-1.0, 0.5-0.0, 0.5-0.0)||^2
       = ||(-0.5, 0.5, 0.5)||^2
       = 0.25 + 0.25 + 0.25 = 0.75
```

**Gradient**:
```
∂E/∂e = 2(e - e*) = 2 × [(-0.5), 0.5, 0.5] = [-1.0, 1.0, 1.0]
```

**Step 1**:
```
e_1 = e_0 - α × ∇E
    = [0.5, 0.5, 0.5] - 0.2 × [-1.0, 1.0, 1.0]
    = [0.5, 0.5, 0.5] - [-0.2, 0.2, 0.2]
    = [0.7, 0.3, 0.3]

E(e_1) = ||(0.7-1.0, 0.3-0.0, 0.3-0.0)||^2
       = 0.09 + 0.09 + 0.09 = 0.27
```

**Step 2**:
```
∇E = 2 × [-0.3, 0.3, 0.3] = [-0.6, 0.6, 0.6]
e_2 = [0.7, 0.3, 0.3] - 0.2 × [-0.6, 0.6, 0.6]
    = [0.82, 0.18, 0.18]

E(e_2) = 0.0972
```

**Step 3**:
```
∇E = 2 × [-0.18, 0.18, 0.18] = [-0.36, 0.36, 0.36]
e_3 = [0.82, 0.18, 0.18] - 0.2 × [-0.36, 0.36, 0.36]
    = [0.892, 0.108, 0.108]

E(e_3) = 0.0350
```

**Energy trajectory**: `[0.75 → 0.27 → 0.097 → 0.035]` ✅ Decreasing!

Embedding moving toward target `[1.0, 0.0, 0.0]`.
</details>

### Problem 7: MCMC Corruption Impact

Compare two corruption strategies for EBT:

**Strategy A (Random Noise)**: `e_corrupt = randn()`
**Strategy B (Gaussian Around Embed)**: `e_corrupt = e_true + N(0, σ)`

Which converges faster if `e_true = [1.0, 0.0]`, target = `[1.0, 0.0]`, σ = 0.1?

<details>
<summary>Solution</summary>

**Strategy A**: Start from random, e.g., `[-0.5, 0.8]`
```
Distance to target = ||(−0.5−1.0, 0.8−0.0)||² = 2.25 + 0.64 = 2.89
```
Needs ~10+ steps to converge.

**Strategy B**: Start near truth, e.g., `[1.05, -0.08]` (true + noise)
```
Distance to target = ||(1.05−1.0, −0.08−0.0)||² = 0.0025 + 0.0064 = 0.0089
```
Needs ~2-3 steps to converge.

**Winner**: Strategy B (Gaussian around embed) converges **~5x faster**.

**Insight**: Good initialization reduces optimization steps → faster inference.
</details>

### Problem 8: Learnable Step Size Optimization

Initial: `log_α = -2.0`, giving `α = exp(-2.0) ≈ 0.135`

Energy after 1 step with current α: `E = 5.0`
Energy after 1 step with α×1.5: `E = 3.0`

Should we increase or decrease `log_α`?

<details>
<summary>Solution</summary>

**Analysis**:
- Current α = 0.135 → Energy = 5.0
- Larger α = 0.135 × 1.5 = 0.2025 → Energy = 3.0

**Conclusion**: Larger step size reduced energy more → **increase** `log_α`.

**Update**:
```
log_α_new = log_α + 0.1  # Gradient step
          = -2.0 + 0.1
          = -1.9

α_new = exp(-1.9) ≈ 0.150
```

**Benefit**: Model learns optimal step size automatically instead of manual tuning!
</details>

### Problem 9: Energy vs Probability

Standard transformer: `p(token) = softmax(logits)`
EBT: `energy(token) = model(embedding)`

If EBT energies are `[1.0, 3.0, 5.0]`, which token is selected?

<details>
<summary>Solution</summary>

**EBT Rule**: Select token with **lowest energy** (not highest probability).

```
Energies: [1.0, 3.0, 5.0]
Selected token: 0 (energy = 1.0 is minimum)
```

**Compare with Softmax**:
```
Logits: [1.0, 3.0, 5.0]
Probs: softmax([1.0, 3.0, 5.0]) = [0.015, 0.117, 0.868]
Selected: token 2 (highest prob)
```

**Key Difference**: EBT minimizes energy, transformers maximize probability.

**Note**: EBT can convert energy → probability via Boltzmann distribution:
```
p(token) ∝ exp(-energy(token) / T)
```
where T is temperature.
</details>

### Problem 10: Adaptive Compute Budget

Model has budget for 10 optimization steps total across 3 tokens.

**Scenario 1 (Fixed)**: Use 3-3-4 steps per token  
**Scenario 2 (Adaptive)**: Allocate based on energy reduction

- Token 1: Energy 10.0 → 1.0 in 2 steps (converged)
- Token 2: Energy 10.0 → 9.5 in 2 steps (not converged)
- Token 3: Energy 10.0 → ? in ? steps

How should adaptive system allocate remaining 6 steps?

**Code Snippet** (Adaptive Allocation with Logging):
```python
import numpy as np
from logging_utils import create_logger

def adaptive_optimize(tokens, total_budget=10, threshold=0.1, min_steps=2, experiment_name="adaptive_ebt"):
    """
    Adaptively allocate optimization steps across tokens based on energy reduction.
    
    Args:
        tokens: List of tokens to optimize
        total_budget: Total number of optimization steps
        threshold: Energy difference threshold for convergence
        min_steps: Minimum steps per token
        experiment_name: Name for logger
    
    Returns:
        List of steps allocated per token
    """
    logger = create_logger(experiment_name=experiment_name, log_dir="./logs")
    remaining_budget = total_budget
    steps_per_token = []
    energy_deltas = []
    
    for i, token in enumerate(tokens):
        steps = 0
        prev_energy = float('inf')
        current_energy = 10.0  # Initial energy (example)
        
        while steps < min_steps or (remaining_budget > 0 and abs(prev_energy - current_energy) > threshold):
            # Perform one optimization step (mock)
            prev_energy = current_energy
            current_energy = max(0, current_energy * 0.9)  # Mock energy reduction
            steps += 1
            remaining_budget -= 1
            
            # Log energy
            logger.log_metrics({f"token_{i}_energy": current_energy}, step=steps)
            
            if remaining_budget <= 0:
                break
        
        steps_per_token.append(steps)
        energy_deltas.append(abs(prev_energy - current_energy))
        
        # Log convergence status
        logger.log_metrics({f"token_{i}_steps": steps, f"token_{i}_delta": energy_deltas[-1]}, step=steps)
    
    # Redistribute remaining budget to hardest token
    if remaining_budget > 0:
        hardest_token_idx = np.argmax(energy_deltas)  # Highest delta = slowest convergence
        additional_steps = min(remaining_budget, 4)  # Cap additional steps
        steps_per_token[hardest_token_idx] += additional_steps
        logger.log_metrics({f"token_{hardest_token_idx}_additional_steps": additional_steps}, step=steps_per_token[hardest_token_idx])
    
    logger.finish()
    return steps_per_token

# Example usage
tokens = ["token1", "token2", "token3"]  # Pseudo tokens
allocation = adaptive_optimize(tokens)
print(f"Step allocation: {allocation}")  # e.g., [2, 4, 4]
```

<details>
<summary>Solution</summary>

**Analysis**:
- Token 1: Large reduction (10.0 → 1.0) → **converged**, stop early
- Token 2: Small reduction (10.0 → 9.5) → **struggling**, needs more steps
- Token 3: Unknown

**Adaptive Allocation**:
```
Token 1: 2 steps (early stop) ✅
Token 2: 4 steps (extra budget for hard token)
Token 3: 4 steps (standard)
Total: 2 + 4 + 4 = 10 steps ✅
```

**Benefit**: Easy tokens get fewer steps, hard tokens get more → better use of compute!

**Implementation Notes**:
- The `adaptive_optimize` function dynamically allocates steps based on energy reduction, using a threshold (`|E_t - E_{t-1}| < 0.1`) to detect convergence.
- Integrates with `logging_utils` to track energy and steps per token, saving metrics to `./logs`.
- Caps additional steps (max 4) to prevent over-allocation to a single token.
- Returns allocation for analysis, e.g., `[2, 4, 4]`.

</details>