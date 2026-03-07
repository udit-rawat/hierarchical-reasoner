"""
Experiment: HRM Depth vs Accuracy
Sweeps num_steps=[1,2,3,4,5] on 4x4 Sudoku and reports cell/puzzle accuracy.

Usage:
    python3 experiments/compare_depth_accuracy.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import HierarchicalReasoningModel
from src.datasetSudoku import SudokuDataset
from src.utils import set_seed

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_DIM = 64
L_ITERATIONS = 5
GRID_SIZE = 2          # 4x4
TRAIN_SAMPLES = 200
TEST_SAMPLES = 50
DEPTH_SWEEP = [1, 2, 3, 4, 5]


def accuracy(preds, targets):
    grid_size = int(preds.shape[1] ** 0.5)  # sqrt(16)=4 for 4x4
    preds = torch.round(preds * grid_size)
    targets = torch.round(targets * grid_size)
    correct = (torch.abs(preds - targets) < 0.5).float()
    cell_acc = correct.mean().item() * 100
    puzzle_acc = (correct.mean(dim=1) == 1.0).float().mean().item() * 100
    return cell_acc, puzzle_acc


def train_and_eval(num_steps, train_loader, test_loader):
    set_seed(SEED)
    input_dim = (GRID_SIZE ** 2) ** 2  # 16

    model = HierarchicalReasoningModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=input_dim,
        num_steps=num_steps,
        l_iterations=L_ITERATIONS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        for puzzles, solutions in train_loader:
            puzzles, solutions = puzzles.to(DEVICE), solutions.to(DEVICE)
            optimizer.zero_grad()
            out = model(puzzles)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(preds, solutions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Final evaluation
    model.eval()
    total_cell, total_puzzle, n = 0, 0, 0
    with torch.no_grad():
        for puzzles, solutions in test_loader:
            puzzles, solutions = puzzles.to(DEVICE), solutions.to(DEVICE)
            out = model(puzzles)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            c, p = accuracy(preds, solutions)
            total_cell += c
            total_puzzle += p
            n += 1

    params = sum(p.numel() for p in model.parameters())
    return total_cell / n, total_puzzle / n, params


def main():
    set_seed(SEED)

    print("=" * 65)
    print("  HRM Depth vs Accuracy — 4×4 Sudoku Sweep")
    print(f"  Device: {DEVICE} | Epochs: {EPOCHS} | L-iters: {L_ITERATIONS}")
    print("=" * 65)

    train_dataset = SudokuDataset(TRAIN_SAMPLES, grid_size=GRID_SIZE, seed=SEED)
    test_dataset  = SudokuDataset(TEST_SAMPLES,  grid_size=GRID_SIZE, seed=SEED + 1)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    results = []
    for num_steps in DEPTH_SWEEP:
        total_depth = num_steps * L_ITERATIONS
        print(f"\n[num_steps={num_steps}] total depth={total_depth} ... ", end='', flush=True)
        t0 = time.time()
        cell_acc, puzzle_acc, params = train_and_eval(num_steps, train_loader, test_loader)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.0f}s)")
        results.append((num_steps, total_depth, cell_acc, puzzle_acc, params, elapsed))

    print("\n" + "=" * 65)
    print(f"  {'Steps':>5}  {'Depth':>5}  {'Params':>8}  {'Cell%':>7}  {'Puzzle%':>8}  {'Time':>6}")
    print("-" * 65)
    for num_steps, depth, cell_acc, puzzle_acc, params, elapsed in results:
        print(f"  {num_steps:>5}  {depth:>5}  {params:>8,}  {cell_acc:>6.1f}%  {puzzle_acc:>7.1f}%  {elapsed:>5.0f}s")
    print("=" * 65)

    best = max(results, key=lambda r: r[3])
    print(f"\n  Best puzzle accuracy: num_steps={best[0]} -> {best[3]:.1f}%")


if __name__ == "__main__":
    main()
