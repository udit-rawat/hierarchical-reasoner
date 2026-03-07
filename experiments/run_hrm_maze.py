"""
Experiment: HRM on Maze Pathfinding
Trains HRM on 5x5 maze dataset and evaluates path-finding accuracy.

Usage:
    python3 experiments/run_hrm_maze.py
    python3 experiments/run_hrm_maze.py --maze_size 7
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import HierarchicalReasoningModel
from src.datasetMaze import MazeDataset
from src.utils import set_seed

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--maze_size',     type=int, default=5)
    p.add_argument('--train_samples', type=int, default=500)
    p.add_argument('--test_samples',  type=int, default=100)
    p.add_argument('--hidden_dim',    type=int, default=64)
    p.add_argument('--num_steps',     type=int, default=3)
    p.add_argument('--l_iterations', type=int, default=5)
    p.add_argument('--epochs',        type=int, default=50)
    p.add_argument('--batch_size',    type=int, default=32)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--seed',          type=int, default=42)
    return p.parse_args()


def cell_accuracy(preds, targets, tol=0.1):
    """% of cells with prediction within tol of target."""
    correct = (torch.abs(preds - targets) < tol).float()
    cell_acc = correct.mean().item() * 100
    # Path accuracy: cells that are on the solution path (value ~0.25)
    path_mask = (targets > 0.1) & (targets < 0.4)
    if path_mask.sum() > 0:
        path_acc = correct[path_mask].mean().item() * 100
    else:
        path_acc = 0.0
    return cell_acc, path_acc


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_cell, total_path = 0, 0, 0
    for mazes, solutions in loader:
        mazes, solutions = mazes.to(DEVICE), solutions.to(DEVICE)
        optimizer.zero_grad()
        out = model(mazes)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        loss = criterion(preds, solutions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        c, p = cell_accuracy(preds.detach(), solutions)
        total_loss += loss.item(); total_cell += c; total_path += p
    n = len(loader)
    return total_loss/n, total_cell/n, total_path/n


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_cell, total_path = 0, 0, 0
    with torch.no_grad():
        for mazes, solutions in loader:
            mazes, solutions = mazes.to(DEVICE), solutions.to(DEVICE)
            out = model(mazes)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(preds, solutions)
            c, p = cell_accuracy(preds, solutions)
            total_loss += loss.item(); total_cell += c; total_path += p
    n = len(loader)
    return total_loss/n, total_cell/n, total_path/n


def main():
    args = parse_args()
    set_seed(args.seed)

    input_dim = args.maze_size ** 2

    print("=" * 65)
    print(f"  HRM on Maze Pathfinding — {args.maze_size}×{args.maze_size} grid")
    print(f"  Device: {DEVICE} | Epochs: {args.epochs} | H-steps: {args.num_steps}")
    print("=" * 65)

    train_ds = MazeDataset(args.train_samples, maze_size=args.maze_size, seed=args.seed)
    test_ds  = MazeDataset(args.test_samples,  maze_size=args.maze_size, seed=args.seed+1)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model = HierarchicalReasoningModel(
        input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=input_dim,
        num_steps=args.num_steps, l_iterations=args.l_iterations
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {params:,} | Input/Output dim: {input_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_path_acc = 0
    print(f"\n  {'Epoch':>5}  {'Loss':>8}  {'Cell%':>7}  {'Path%':>7}  {'ValLoss':>8}  {'ValCell%':>9}  {'ValPath%':>9}")
    print("  " + "-" * 65)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_cell, tr_path = train_epoch(model, train_dl, optimizer, criterion)
        va_loss, va_cell, va_path = evaluate(model, test_dl, criterion)

        if va_path > best_path_acc:
            best_path_acc = va_path

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>5}  {tr_loss:>8.4f}  {tr_cell:>6.1f}%  {tr_path:>6.1f}%  "
                  f"{va_loss:>8.4f}  {va_cell:>8.1f}%  {va_path:>8.1f}%")

    print("\n" + "=" * 65)
    print(f"  Best path accuracy: {best_path_acc:.1f}%")

    # Show a sample prediction
    print("\n  Sample maze vs prediction:")
    train_ds.visualize_sample(0)

    print("\n  Summary:")
    print(f"  Task       : {args.maze_size}×{args.maze_size} maze pathfinding")
    print(f"  Model      : HRM (H-steps={args.num_steps}, L-iters={args.l_iterations})")
    print(f"  Cell acc   : {va_cell:.1f}%  (all cells correct)")
    print(f"  Path acc   : {va_path:.1f}%  (path cells correct)")
    print(f"  Best path  : {best_path_acc:.1f}%")


if __name__ == "__main__":
    main()
