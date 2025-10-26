# Hierarchical Reasoning Model (HRM) - Implementation Summary

## 📋 Project Overview

Implementation of "Hierarchical Reasoning Model" (arXiv:2506.21734) - A paper-aligned neural architecture with multi-timescale hierarchical processing.

---

## ✅ COMPLETED WORK

### **Phase 1: Core Implementation (DONE)**

#### **1. `src/model.py` ✅**

**Status:** Paper-aligned implementation complete

**What was implemented:**

- ✅ **HighLevelReasoner (H-module)**: Slow planning module with GRU
- ✅ **LowLevelExecutor (L-module)**: Fast execution module with **multi-iteration processing**
- ✅ **ACTModule**: Adaptive Computation Time using Q-learning (optional)
- ✅ **HierarchicalReasoningModel**: Main model combining H and L modules

**Key Innovation Implemented:**

```python
# PAPER'S CORE CONCEPT: Multi-iteration L-module
# L-module runs 5 iterations per H-step (not 1:1 like standard RNNs)
# Example: 3 H-steps × 5 L-iterations = 15 total computations

model = HierarchicalReasoningModel(
    input_dim=1,
    hidden_dim=32,
    output_dim=1,
    num_steps=3,        # H-module steps
    l_iterations=5,     # L-iterations per H-step (PAPER'S KEY FEATURE)
    use_act=False       # Optional adaptive halting
)
```

**What Changed from Original:**
| Component | Before | After (Paper-Aligned) |
|-----------|--------|----------------------|
| L-module iterations | 1 per H-step | **5 per H-step** |
| Reset mechanism | ❌ None | ✅ Resets between H-steps |
| Processing ratio | 1:1 (H:L) | **1:5 (H:L)** |
| Trajectory tracking | ❌ None | ✅ Full reasoning trace |
| ACT module | ❌ None | ✅ Optional Q-learning halting |

---

#### **2. `src/train.py` ✅**

**Status:** Training with paper-aligned features

**What was implemented:**

- ✅ Training loop supporting new `l_iterations` parameter
- ✅ Gradient clipping for stability
- ✅ Optional learning rate scheduling
- ✅ Trajectory logging for analysis
- ✅ Enhanced checkpoint saving (full training state)
- ✅ Training time tracking and metrics

**Key Features:**

```python
trainer = Trainer(
    input_dim=1,
    hidden_dim=32,
    output_dim=1,
    num_steps=3,
    l_iterations=5,      # NEW: Paper-aligned multi-iteration
    use_act=False,       # NEW: Optional ACT
    lr=1e-3,
    epochs=50,
    use_scheduler=False, # NEW: Optional LR scheduling
    log_trajectory=True  # NEW: Trajectory analysis
)
```

**Outputs:**

- Training/validation loss per epoch
- Computation depth analysis (H-steps × L-iterations)
- Best model checkpointing with full state
- Training time metrics

---

#### **3. `src/evaluate.py` ✅**

**Status:** Comprehensive evaluation with trajectory analysis

**What was implemented:**

- ✅ Load checkpoints with `strict=False` (handles ACT module mismatches)
- ✅ Comprehensive metrics (MSE, MAE, RMSE, Correlation, R²)
- ✅ Trajectory analysis and visualization
- ✅ Per-step reasoning breakdown
- ✅ L-module convergence tracking
- ✅ Reasoning depth comparison

**Key Features:**

```python
evaluator = Evaluator(
    checkpoint_path="hrm_checkpoint.pt",
    input_dim=1,
    hidden_dim=32,
    output_dim=1,
    num_steps=3,
    l_iterations=5,  # MUST match training config!
    use_act=False    # MUST match training config!
)

# Comprehensive evaluation
results = evaluator.evaluate(verbose=True, analyze_trajectory=True)

# Sample-level trajectory analysis
pred, trajectory = evaluator.evaluate_sample(input, show_trajectory=True)
```

**Outputs:**

- Standard metrics (MSE, MAE, RMSE, Correlation, R²)
- Hierarchical reasoning analysis (H-steps, L-iterations, depth)
- Step-by-step trajectory breakdown
- L-module convergence visualization

---

#### **4. `src/dataset.py` ✅**

**Status:** Basic synthetic arithmetic dataset

**Current Implementation:**

- Simple add/subtract tasks
- Input: `[a, b, op]` → Output: `[a+b]` or `[a-b]`
- 8000 training samples, 2000 test samples

**Note:** This is minimal but functional. Can be enhanced later.

---

## 🔴 KNOWN ISSUES & FIXES APPLIED

### **Issue 1: Indentation Errors (FIXED ✅)**

- **Problem:** Inconsistent indentation in original code
- **Fix:** Applied consistent 4-space indentation across all files

### **Issue 2: Dimension Mismatch (FIXED ✅)**

- **Problem:** GRU expected `input_size=1` but got `32`
- **Root Cause:** L-module output fed back into H-module without projection
- **Fix:** Added `input_projection` layer in HighLevelReasoner

### **Issue 3: Checkpoint Loading Error (FIXED ✅)**

- **Problem:** `RuntimeError: Unexpected key(s) in state_dict: "act_module..."`
- **Root Cause:** Training with `use_act=True`, evaluating with `use_act=False`
- **Fix:** Changed `load_state_dict(strict=False)` to allow mismatches

---

## 📊 PAPER ALIGNMENT STATUS

### **Core Concepts from Paper:**

| Paper Component                     | Implementation Status | Notes                                        |
| ----------------------------------- | --------------------- | -------------------------------------------- |
| **Hierarchical Convergence**        | ✅ IMPLEMENTED        | L-module runs 5 iterations per H-step        |
| **L-module Reset**                  | ✅ IMPLEMENTED        | Resets between H-steps with control signal   |
| **Multi-timescale Processing**      | ✅ IMPLEMENTED        | 1:5 ratio (slow H, fast L)                   |
| **ACT (Q-learning halting)**        | ✅ IMPLEMENTED        | Optional, currently disabled                 |
| **One-step Gradient Approximation** | ❌ NOT IMPLEMENTED    | Uses standard backprop (good enough for now) |
| **Complex Reasoning Tasks**         | ⚠️ PARTIALLY          | Simple arithmetic only, needs Sudoku/Maze    |

---

## 🎯 WHAT NEEDS TO BE DONE NEXT

### **Priority 1: Enhanced Datasets (CRITICAL)**

The current dataset (simple arithmetic) doesn't showcase HRM's strengths. Need to implement:

#### **A. Sudoku Dataset (`src/dataset_sudoku.py`)**

```python
# What to implement:
- 4×4 or 9×9 Sudoku puzzle generation
- Input: Partially filled grid
- Output: Complete solution
- Train/test split with varying difficulty
```

**Why important:** Paper shows near-perfect Sudoku solving. This validates hierarchical reasoning.

#### **B. Maze Dataset (`src/dataset_maze.py`)**

```python
# What to implement:
- Grid-based maze generation
- Input: Maze layout + start/goal positions
- Output: Optimal path
- Various maze sizes and complexities
```

**Why important:** Paper demonstrates optimal path finding. Tests multi-step planning.

---

### **Priority 2: Remaining Files**

#### **A. `src/utils.py` (NEEDED)**

```python
# What to implement:
def set_seed(seed):
    """Set random seeds for reproducibility"""

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Enhanced checkpoint saving"""

def load_checkpoint(path, model, optimizer=None):
    """Enhanced checkpoint loading"""

def log_metrics(metrics, writer=None):
    """Log metrics to console/tensorboard"""

def visualize_trajectory(trajectory, save_path=None):
    """Visualize reasoning trajectory"""
```

#### **B. `experiments/run_hrm_sudoku.py` (NEEDED)**

```python
# What to implement:
- Load Sudoku dataset
- Train HRM on Sudoku
- Evaluate and report accuracy
- Compare with baseline RNN
```

#### **C. `experiments/run_baseline_rnn.py` (OPTIONAL)**

```python
# What to implement:
- Standard RNN/LSTM baseline
- Train on same tasks
- Compare performance vs HRM
```

#### **D. Configuration Files (RECOMMENDED)**

**`configs/model_config.yaml`:**

```yaml
model:
  input_dim: 1
  hidden_dim: 32
  output_dim: 1
  num_steps: 3
  l_iterations: 5
  use_act: false
```

**`configs/train_config.yaml`:**

```yaml
training:
  batch_size: 32
  epochs: 50
  lr: 0.001
  use_scheduler: false
  log_trajectory: true
```

#### **E. `notebooks/sanity_check.ipynb` (RECOMMENDED)**

- Quick model testing
- Trajectory visualization
- Hyperparameter experiments

#### **F. Documentation**

**`README.md`:**

- Project overview
- Installation instructions
- How to run experiments
- Results and comparisons

**`theory_notes.md`:**

- Paper summary
- Mathematical intuition
- Architecture diagrams
- Design decisions

**`requirements.txt`:**

```txt
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

### **Priority 3: Enhancements (OPTIONAL)**

#### **A. ACT Training**

Currently ACT module is implemented but not trained. Need to:

- Implement Q-learning reward signal
- Train ACT to decide halt/continue
- Compare fixed vs adaptive computation

#### **B. One-Step Gradient Approximation**

Paper uses special training technique to avoid BPTT:

- Implement gradient detachment between H-steps
- Compare with standard backprop
- Measure training speed improvement

#### **C. Visualization Tools**

- Plot reasoning trajectories
- Visualize H-module planning
- Show L-module convergence
- Compare shallow vs deep reasoning

---

## 🔧 TESTING COMMANDS

### **Current Working Commands:**

```bash
# Test model architecture
python3 src/model.py

# Train model
python3 src/train.py

# Evaluate model
python3 src/evaluate.py
```

### **Expected Output Structure:**

```
HierarchicalReasoningModel/
├── src/
│   ├── model.py ✅ DONE
│   ├── train.py ✅ DONE
│   ├── dataset.py ✅ DONE (basic)
│   ├── evaluate.py ✅ DONE
│   ├── utils.py ❌ TODO
│   ├── dataset_sudoku.py ❌ TODO
│   └── dataset_maze.py ❌ TODO
│
├── experiments/
│   ├── run_hrm_sudoku.py ❌ TODO
│   └── run_baseline_rnn.py ❌ TODO (optional)
│
├── configs/
│   ├── model_config.yaml ❌ TODO
│   └── train_config.yaml ❌ TODO
│
├── notebooks/
│   └── sanity_check.ipynb ❌ TODO
│
├── requirements.txt ❌ TODO
├── README.md ❌ TODO
└── theory_notes.md ❌ TODO
```

---

## 🎓 KEY LEARNINGS

### **Paper's Core Innovation:**

The key insight is **hierarchical convergence** with **multi-timescale processing**:

- H-module: Slow strategic planning (updates every 5 steps)
- L-module: Fast tactical execution (runs 5 iterations per H-update)
- Total computation: 3 H-steps × 5 L-iterations = 15 computational steps

This mimics human cognitive hierarchy:

- **System 2 (H-module)**: Slow, deliberate, strategic
- **System 1 (L-module)**: Fast, automatic, tactical

### **Why This Matters:**

Standard RNNs process sequentially at one timescale. HRM adds depth without adding sequence length:

- More computational depth = better reasoning
- Hierarchical structure = more efficient than flat processing
- Learned planning (H-module) + learned execution (L-module)

---

## 📝 HANDOFF CHECKLIST

### **What's Working:**

- ✅ Paper-aligned model architecture
- ✅ Multi-iteration L-module (5 iterations per H-step)
- ✅ L-module reset between H-steps
- ✅ Training loop with trajectory logging
- ✅ Comprehensive evaluation with metrics
- ✅ Checkpoint loading with ACT flexibility

### **What's Tested:**

- ✅ Model forward pass
- ✅ Training on simple arithmetic
- ✅ Evaluation with trajectory analysis
- ✅ Checkpoint save/load

### **What Needs Attention:**

1. **CRITICAL:** Implement Sudoku/Maze datasets to validate paper claims
2. **IMPORTANT:** Create `utils.py` for better code organization
3. **RECOMMENDED:** Add configuration files for reproducibility
4. **OPTIONAL:** Train ACT module, implement one-step gradients

### **Commands for Next Session:**

```bash
# Quick test everything works
python3 src/model.py && python3 src/train.py && python3 src/evaluate.py

# Start with Sudoku dataset
# Create src/dataset_sudoku.py based on paper description
```

---

## 📚 References

**Paper:** "Hierarchical Reasoning Model" - arXiv:2506.21734

- Near-perfect Sudoku solving with 27M parameters
- Optimal maze path finding
- Outperforms larger models on ARC benchmark
- Only 1000 training samples needed

**Key Quote from Paper:**

> "HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations."

---

## 🚀 QUICK START FOR NEXT SESSION

1. **Verify current setup works:**

   ```bash
   python3 src/model.py
   ```

2. **Create Sudoku dataset next:**

   - Start with 4×4 Sudoku (simpler)
   - Generate training data (1000 samples as per paper)
   - Ensure input/output format works with HRM

3. **Update `dataset.py` or create new file:**

   ```python
   # src/dataset_sudoku.py
   class SudokuDataset(Dataset):
       # Implement puzzle generation
       # Paper uses constraint satisfaction for generation
   ```

4. **Train on Sudoku:**
   ```python
   trainer = Trainer(
       input_dim=81,  # 9×9 grid
       hidden_dim=128,  # Larger for complex task
       output_dim=81,
       num_steps=5,  # More steps for complex reasoning
       l_iterations=10  # More L-iterations
   )
   ```

---

**END OF SUMMARY**

_All core components are paper-aligned and functional. Next phase: validation on complex reasoning tasks (Sudoku/Maze) as described in the paper._
