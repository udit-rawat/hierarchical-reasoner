"""
Experiment: Visualize HRM Reasoning Trajectories
Plots H-step control signals and L-module activations across reasoning steps.

Usage:
    python3 experiments/visualize_trajectories.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.model import HierarchicalReasoningModel
from src.datasetSudoku import SudokuDataset
from src.utils import set_seed

set_seed(42)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Build and briefly train a 4x4 model so trajectories are non-trivial
import torch.nn as nn
from torch.utils.data import DataLoader

INPUT_DIM = 16
HIDDEN_DIM = 64
NUM_STEPS = 4
L_ITERS = 5

model = HierarchicalReasoningModel(
    input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM,
    num_steps=NUM_STEPS, l_iterations=L_ITERS
).to(DEVICE)

# Quick training pass so weights aren't random noise
dataset = SudokuDataset(300, grid_size=2, seed=42)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)
opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
crit    = nn.MSELoss()

print("Training 40 epochs for non-trivial trajectories...")
for epoch in range(40):
    model.train()
    for puzzles, solutions in loader:
        puzzles, solutions = puzzles.to(DEVICE), solutions.to(DEVICE)
        opt.zero_grad()
        out = model(puzzles)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        crit(preds, solutions).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

# Pick 3 sample puzzles
samples = [dataset[i] for i in range(3)]
fig = plt.figure(figsize=(18, 12))
fig.suptitle("HRM Reasoning Trajectories — 4×4 Sudoku", fontsize=16, fontweight='bold', y=0.98)

for sample_idx, (puzzle, solution) in enumerate(samples):
    model.eval()
    with torch.no_grad():
        inp = puzzle.unsqueeze(0).to(DEVICE)
        _, traj = model(inp, return_trajectory=True)

    num_h = traj['num_h_steps']   # 4
    l_per_h = traj['l_iterations_per_step']  # 5

    # ── Row 1: control signal heatmap per H-step ──────────────────────────────
    ax_ctrl = fig.add_subplot(3, 3, sample_idx + 1)
    ctrl_matrix = np.stack(
        [step['control_signal'].squeeze(0).cpu().numpy() for step in traj['steps']]
    )  # [num_h, hidden_dim]
    im = ax_ctrl.imshow(ctrl_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax_ctrl.set_title(f"Sample {sample_idx+1} — Control Signals\n(H-steps × hidden_dim)", fontsize=10)
    ax_ctrl.set_xlabel("Hidden dim", fontsize=8)
    ax_ctrl.set_ylabel("H-step", fontsize=8)
    ax_ctrl.set_yticks(range(num_h))
    ax_ctrl.set_yticklabels([f"H{i}" for i in range(num_h)], fontsize=8)
    plt.colorbar(im, ax=ax_ctrl, fraction=0.046, pad=0.04)

    # ── Row 2: L-module trajectory norm across iterations ─────────────────────
    ax_l = fig.add_subplot(3, 3, sample_idx + 4)
    for h_idx, step in enumerate(traj['steps']):
        l_traj = step['l_trajectory'].squeeze(0).cpu().numpy()  # [l_iters, hidden]
        norms = np.linalg.norm(l_traj, axis=1)
        ax_l.plot(range(l_per_h), norms, marker='o', markersize=4,
                  label=f"H{h_idx}", linewidth=1.5)
    ax_l.set_title(f"Sample {sample_idx+1} — L-module Convergence\n(activation norm per iteration)", fontsize=10)
    ax_l.set_xlabel("L-iteration", fontsize=8)
    ax_l.set_ylabel("||activation||", fontsize=8)
    ax_l.legend(fontsize=7, loc='upper right')
    ax_l.grid(True, alpha=0.3)

    # ── Row 3: final L-state delta between H-steps ────────────────────────────
    ax_d = fig.add_subplot(3, 3, sample_idx + 7)
    states = [step['final_l_state'].squeeze(0).cpu().numpy() for step in traj['steps']]
    deltas = [np.linalg.norm(states[i+1] - states[i]) for i in range(len(states)-1)]
    ax_d.bar(range(len(deltas)), deltas, color='steelblue', alpha=0.8)
    ax_d.set_title(f"Sample {sample_idx+1} — State Delta Between H-steps\n(||L_state[t+1] - L_state[t]||)", fontsize=10)
    ax_d.set_xlabel("H-step transition", fontsize=8)
    ax_d.set_ylabel("Delta norm", fontsize=8)
    ax_d.set_xticks(range(len(deltas)))
    ax_d.set_xticklabels([f"H{i}→H{i+1}" for i in range(len(deltas))], fontsize=8)
    ax_d.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = "assets/reasoning_trajectories.png"
os.makedirs("assets", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()
