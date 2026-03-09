# HRM — Theory Notes & Experiment Findings

Paper: **Hierarchical Reasoning Model** (arXiv:2506.21734) — Guan Wang et al., Sapient Intelligence, June 2025.

---

## Core Idea

HRM solves complex reasoning tasks in a **single forward pass** using two interdependent recurrent modules operating at different timescales:

```
Input → H-module (slow, 1×) → control signal
                                    → L-module (fast, 5×) → converged state
                                                                  → H-module (next step) ...
                                                                                          → Output
```

| Module | Timescale | Role |
|---|---|---|
| H-module (HighLevelReasoner) | Slow — 1× per step | Strategic planner, GRU + control signal |
| L-module (LowLevelExecutor) | Fast — 5× per H-step | Tactical executor, GRU + refinement |
| ACT | Optional | Q-learning halt/continue decision |

Analogy to Kahneman's dual-process theory:
- **H-module = System 2** — slow, deliberate, strategic
- **L-module = System 1** — fast, automatic, tactical

---

## Implementation

### Architecture
- `HighLevelReasoner` — GRU with control signal projection
- `LowLevelExecutor` — GRU running `l_iterations` per H-step, resets each step
- `ACTModule` — Q-network (2 actions: halt=0, continue=1)
- `HierarchicalReasoningModel` — combines above, optional one-step gradient

### Key flags
```python
HierarchicalReasoningModel(
    input_dim=81,
    hidden_dim=128,
    output_dim=81,
    num_steps=5,         # H-module iterations
    l_iterations=10,     # L-iterations per H-step
    use_act=False,       # adaptive halting
    one_step_grad=False, # paper's gradient approximation
    min_h_steps=1        # ACT minimum steps
)
```

---

## Experiment Results

### 1. Arithmetic — HRM vs RNN Baseline
- RNN (7,489 params): MSE 0.0091
- HRM (17,121 params): MSE 0.0186
- **Finding:** RNN wins on simple arithmetic. HRM's advantage emerges on tasks with hierarchical structure.

### 2. Gradient Approximation — BPTT vs One-Step
- BPTT: MSE 0.0417, 900s training
- One-Step (paper's method): MSE 0.0654, 647s training
- **Finding:** One-step is 1.39× faster. BPTT achieves better accuracy. Paper's tradeoff confirmed.

### 3. ACT Module (Q-learning halting)
- Fixed depth: always 3 H-steps
- ACT adaptive: halts at step 1 from epoch 7 onwards — **66.7% fewer steps**
- **Finding:** ACT learns to halt early. Accuracy tradeoff with frozen HRM weights expected.

### 4. Depth vs Accuracy (4×4 Sudoku sweep)
| num_steps | Cell Acc |
|---|---|
| 1 | 37.6% |
| 2 | 35.2% |
| 3 | 33.4% |
| 5 | 30.7% |
- **Finding:** Shallower HRM wins on small tasks. More H-steps hurt at 4×4 scale — over-parameterized.

### 5. Maze Pathfinding (5×5)
- 500 training samples, 50 epochs
- Epoch 1: 39.1% path accuracy
- Epoch 20: **100% path accuracy**
- **Finding:** HRM achieves perfect path-finding by epoch 20. Hierarchical reasoning naturally fits the coarse-to-fine structure of maze solving.

### 6. Inference Speed
| Model | ms/sample (batch=32) |
|---|---|
| RNN | 0.029 |
| HRM 1-step | 0.195 |
| HRM 3-step | 0.568 |
| HRM 5-step | 0.940 |
- **Finding:** Cost scales linearly with num_steps. At batch=128, HRM 3-step = 0.15ms/sample.

### 7. Reasoning Trajectories
- Control signals: Each H-step generates a distinct pattern — H-module produces different strategic guidance per step
- L-module convergence: Norms plateau within 5 iterations confirming local convergence
- State deltas: Largest change at H0→H1, progressively smaller — model refines, doesn't rebuild

### 8. 9×9 Sudoku — Negative Result
- 500 epochs, 5000 samples, CrossEntropy loss, 512 hidden_dim
- Val cell accuracy: 15.9% (random = 11.1%)
- Puzzle accuracy: 0%
- **Finding:** Flat GRU hidden state cannot encode per-cell constraint relationships (rows, cols, boxes). The paper likely uses attention over cell states or explicit constraint message passing. This is an open architectural gap — our HRM works for tasks with global hidden state reasoning but not per-cell constraint satisfaction.

---

## What Works vs What Doesn't

| Task | Result | Why |
|---|---|---|
| Arithmetic | ✅ Learns | Single output, global reasoning |
| Maze 5×5 | ✅ 100% | Path = sequential decision, fits GRU |
| 4×4 Sudoku (MSE) | ⚠️ Partial | Too small to matter |
| 9×9 Sudoku | ❌ Fails | Per-cell constraint propagation needed |

---

## Open Questions

1. Would a cell-attention output head (81 separate attention-based decoders) solve 9×9 Sudoku?
2. Does ACT accuracy improve with joint HRM+ACT training?
3. How does HRM scale to ARC-AGI tasks — do they require per-element reasoning too?

---

## References

- Paper: https://arxiv.org/abs/2506.21734
- Guan Wang et al., Sapient Intelligence, June 2025
