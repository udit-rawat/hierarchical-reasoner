# hierarchical-reasoner

experiment reproducible of the Hierarchical Reasoning Model (HRM) by Sapient Intelligence (Guan Wang et al., June 2025) in pytorch.

# Directory Structure

HierarchicalReasoningModel/
│
├── src/
│ ├── **init**.py
│ ├── model.py # HRM architecture: high-level + low-level modules
│ ├── train.py # Training loop (forward, loss, backward, optimizer)
│ ├── dataset.py # Synthetic datasets (Sudoku, Maze, or dummy reasoning)
│ ├── evaluate.py # Evaluation script (accuracy, qualitative results)
│ ├── utils.py # Helper functions (seed setting, logging, checkpointing)
│
├── experiments/
│ ├── run_hrm_sudoku.py # Simple experiment script (loads dataset, trains HRM)
│ ├── run_baseline_rnn.py # Baseline comparison (optional later)
│
├── configs/
│ ├── model_config.yaml # Hyperparameters (hidden dims, layers, etc.)
│ ├── train_config.yaml # Training params (epochs, batch_size, lr, etc.)
│
├── notebooks/
│ ├── sanity_check.ipynb # Quick notebook for debugging model outputs
│
├── data/ # (Optional) local datasets if needed
│
├── requirements.txt # torch, numpy, matplotlib only
├── README.md # Project overview, how to run
└── theory_notes.md # Paper interpretation, math intuition
