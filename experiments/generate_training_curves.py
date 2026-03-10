"""
Generate training curves for README — runs maze experiment and saves plots.
Usage: python3 experiments/generate_training_curves.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.model import HierarchicalReasoningModel
from src.datasetMaze import MazeDataset
from src.utils import set_seed

set_seed(42)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
EPOCHS = 50


def cell_accuracy(preds, targets, tol=0.1):
    correct = (torch.abs(preds - targets) < tol).float()
    cell_acc = correct.mean().item() * 100
    path_mask = (targets > 0.1) & (targets < 0.4)
    path_acc = correct[path_mask].mean().item() * 100 if path_mask.sum() > 0 else 0.0
    return cell_acc, path_acc


print("Training maze experiment for curves...")
train_ds = MazeDataset(500, maze_size=5, seed=42)
test_ds  = MazeDataset(100, maze_size=5, seed=43)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False)

model = HierarchicalReasoningModel(
    input_dim=25, hidden_dim=64, output_dim=25,
    num_steps=3, l_iterations=5
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

history = {'tr_loss': [], 'va_loss': [], 'tr_cell': [], 'va_cell': [],
           'tr_path': [], 'va_path': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    tl, tc, tp, n = 0, 0, 0, 0
    for mazes, solutions in train_dl:
        mazes, solutions = mazes.to(DEVICE), solutions.to(DEVICE)
        optimizer.zero_grad()
        out = model(mazes)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        loss = criterion(preds, solutions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        c, p = cell_accuracy(preds.detach(), solutions)
        tl += loss.item(); tc += c; tp += p; n += 1
    history['tr_loss'].append(tl/n); history['tr_cell'].append(tc/n); history['tr_path'].append(tp/n)

    model.eval()
    vl, vc, vp, n = 0, 0, 0, 0
    with torch.no_grad():
        for mazes, solutions in test_dl:
            mazes, solutions = mazes.to(DEVICE), solutions.to(DEVICE)
            out = model(mazes)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(preds, solutions)
            c, p = cell_accuracy(preds, solutions)
            vl += loss.item(); vc += c; vp += p; n += 1
    history['va_loss'].append(vl/n); history['va_cell'].append(vc/n); history['va_path'].append(vp/n)

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:>3} | val_cell={vc/n:.1f}% path={vp/n:.1f}%")

# Plot
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("HRM Training Curves — 5×5 Maze Pathfinding", fontsize=14, fontweight='bold')

axes[0].plot(epochs, history['tr_loss'], label='Train', linewidth=2)
axes[0].plot(epochs, history['va_loss'], label='Val',   linewidth=2)
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs, history['tr_cell'], label='Train', linewidth=2)
axes[1].plot(epochs, history['va_cell'], label='Val',   linewidth=2)
axes[1].axhline(100, color='green', linestyle='--', alpha=0.5, label='100%')
axes[1].set_title('Cell Accuracy (%)'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(epochs, history['tr_path'], label='Train', linewidth=2, color='orange')
axes[2].plot(epochs, history['va_path'], label='Val',   linewidth=2, color='red')
axes[2].axhline(100, color='green', linestyle='--', alpha=0.5, label='100%')
axes[2].set_title('Path Accuracy (%)'); axes[2].set_xlabel('Epoch'); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
os.makedirs('assets', exist_ok=True)
out_path = 'assets/training_curves_maze.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → {out_path}")
