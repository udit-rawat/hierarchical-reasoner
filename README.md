# hierarchical-reasoner

PyTorch reproduction of **[Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2506.21734)** — Guan Wang et al., Sapient Intelligence (June 2025).

HRM solves complex reasoning tasks (Sudoku, Maze, ARC) in a **single forward pass** using two interdependent recurrent modules: a slow high-level planner and a fast low-level executor, without explicit supervision of intermediate steps.

---

## How It Works

The model operates at two timescales, inspired by cognitive hierarchy:

```
Input
  └─> H-module (slow planner, GRU) ──> control signal
                                              └─> L-module (fast executor) × 5 iterations ──> converged state
                                                                                                      └─> H-module (next step) ...
                                                                                                                                  └─> Output
```

| Module | Role | Speed | Mechanism |
|---|---|---|---|
| **H-module** | Strategic planner | Slow (1× per step) | GRU + control signal |
| **L-module** | Tactical executor | Fast (5× per H-step) | GRU + refinement network, resets each H-step |
| **ACT** | Adaptive halting | Optional | Q-learning halt/continue decision |

Default config: **3 H-steps × 5 L-iterations = 15 total computational steps** per forward pass.

---

## Project Status

| Component | Status |
|---|---|
| Model architecture (`src/model.py`) | Done |
| Training pipeline (`src/train.py`) | Done |
| Evaluation + trajectory analysis (`src/evaluate.py`) | Done |
| Arithmetic dataset (`src/dataset.py`) | Done |
| Sudoku dataset — 4×4 and 9×9 (`src/datasetSudoku.py`) | Done |
| Sudoku experiment (`experiments/run_hrm_sudoku.py`) | Done |
| Maze dataset | Not implemented |
| Baseline RNN comparison | Not implemented |
| ACT module training | Implemented, not trained |
| One-step gradient approximation | Not implemented |

---

## Installation

```bash
git clone https://github.com/udit-rawat/hierarchical-reasoner.git
cd hierarchical-reasoner
pip install torch numpy matplotlib tqdm py-sudoku pyyaml
```

---

## Usage

### Test the model architecture
```bash
python3 src/model.py
```

### Train on arithmetic (quick sanity check)
```bash
python3 src/train.py
```

### Train on Sudoku (validates paper's core claim)
```bash
# 9×9 Sudoku, 1000 samples (paper config)
python3 experiments/run_hrm_sudoku.py

# Quick 4×4 test
python3 experiments/run_hrm_sudoku.py --quick_test

# Custom config
python3 experiments/run_hrm_sudoku.py \
  --grid_size 3 \
  --train_samples 1000 \
  --hidden_dim 128 \
  --num_steps 5 \
  --l_iterations 10 \
  --epochs 100
```

### Evaluate a saved checkpoint
```bash
python3 src/evaluate.py
```

---

## Model Config Reference

| Parameter | Arithmetic | Sudoku 9×9 | Description |
|---|---|---|---|
| `input_dim` | 1 | 81 | Flattened input size |
| `hidden_dim` | 32 | 128 | GRU hidden size |
| `output_dim` | 1 | 81 | Output size |
| `num_steps` | 3 | 5 | H-module iterations |
| `l_iterations` | 5 | 10 | L-module iterations per H-step |
| `use_act` | False | False | Adaptive computation time |

---

## Paper Claims Being Validated

- Near-perfect 9×9 Sudoku solving with **~27M parameters** and only **1000 training samples**
- Optimal maze path finding
- Outperforms larger models on ARC benchmark

---

## Directory Structure

```
hierarchical-reasoner/
├── src/
│   ├── model.py            # H-module, L-module, ACT, HierarchicalReasoningModel
│   ├── train.py            # Training loop with trajectory logging
│   ├── dataset.py          # Synthetic arithmetic dataset
│   ├── datasetSudoku.py    # Sudoku puzzle generation (4×4 and 9×9)
│   ├── evaluate.py         # Evaluation: metrics + reasoning trajectory analysis
│   └── utils.py            # Config, checkpointing, logging, visualization
├── experiments/
│   └── run_hrm_sudoku.py   # Full Sudoku training + evaluation experiment
├── theory_notes.md         # Paper interpretation, math intuition, implementation decisions
└── README.md
```

---

## References

- **Paper:** [Hierarchical Reasoning Model — arXiv:2506.21734](https://arxiv.org/abs/2506.21734)
- Guan Wang et al., Sapient Intelligence, June 2025
