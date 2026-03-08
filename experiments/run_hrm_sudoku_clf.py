"""
Experiment: HRM on 9x9 Sudoku — Classification (CrossEntropy)
Treats each cell as 9-class classification instead of MSE regression.

Key change from run_hrm_sudoku.py:
- Output head: hidden_dim -> 81*9 logits (9 classes per cell)
- Loss: CrossEntropyLoss instead of MSE
- Accuracy: argmax over 9 classes per cell

Usage:
    python3 experiments/run_hrm_sudoku_clf.py              # full run
    python3 experiments/run_hrm_sudoku_clf.py --quick_test # 4x4 fast test
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import HierarchicalReasoningModel
from src.datasetSudoku import SudokuDataset
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--grid_size',     type=int,   default=3)
    p.add_argument('--train_samples', type=int,   default=1000)
    p.add_argument('--test_samples',  type=int,   default=200)
    p.add_argument('--difficulty',    type=float, default=0.5)
    p.add_argument('--hidden_dim',    type=int,   default=256)
    p.add_argument('--num_steps',     type=int,   default=5)
    p.add_argument('--l_iterations', type=int,   default=10)
    p.add_argument('--epochs',        type=int,   default=200)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--quick_test',    action='store_true')
    return p.parse_args()


def accuracy(logits, targets):
    """
    logits:  [batch, num_cells*num_classes] — raw model output
    targets: [batch, num_cells] — class indices 0-8
    """
    num_cells = targets.shape[1]
    num_classes = logits.shape[1] // num_cells
    logits = logits.view(-1, num_classes, num_cells)   # [batch, 9, 81]
    preds = logits.argmax(dim=1)                        # [batch, 81]

    correct = (preds == targets).float()
    cell_acc = correct.mean().item() * 100
    puzzle_acc = (correct.mean(dim=1) == 1.0).float().mean().item() * 100
    return cell_acc, puzzle_acc


def train_epoch(model, loader, optimizer, criterion, device, num_cells, num_classes):
    model.train()
    total_loss, total_cell, total_puzzle = 0, 0, 0
    pbar = tqdm(loader, leave=False, desc="  batches")
    for puzzles, targets in pbar:
        puzzles, targets = puzzles.to(device), targets.to(device)
        optimizer.zero_grad()

        out = model(puzzles)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        # logits: [batch, num_cells*num_classes] -> [batch, num_classes, num_cells]
        logits_reshaped = logits.view(-1, num_classes, num_cells)

        loss = criterion(logits_reshaped, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        c, p = accuracy(logits.detach(), targets)
        total_loss += loss.item(); total_cell += c; total_puzzle += p
        pbar.set_postfix(loss=f"{loss.item():.4f}", cell=f"{c:.1f}%", puzzle=f"{p:.1f}%")

    n = len(loader)
    return total_loss/n, total_cell/n, total_puzzle/n


def evaluate(model, loader, criterion, device, num_cells, num_classes):
    model.eval()
    total_loss, total_cell, total_puzzle = 0, 0, 0
    with torch.no_grad():
        for puzzles, targets in loader:
            puzzles, targets = puzzles.to(device), targets.to(device)
            out = model(puzzles)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits_reshaped = logits.view(-1, num_classes, num_cells)
            loss = criterion(logits_reshaped, targets)
            c, p = accuracy(logits, targets)
            total_loss += loss.item(); total_cell += c; total_puzzle += p
    n = len(loader)
    return total_loss/n, total_cell/n, total_puzzle/n


def main():
    args = parse_args()

    if args.quick_test:
        print("QUICK TEST MODE (4x4)")
        args.grid_size = 2
        args.train_samples = 100
        args.test_samples = 20
        args.epochs = 30
        args.hidden_dim = 64

    set_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    num_cells = (args.grid_size ** 2) ** 2   # 81 for 9x9
    num_classes = args.grid_size ** 2         # 9 for 9x9
    output_dim = num_cells * num_classes      # 729 for 9x9

    print("=" * 65)
    print(f"  HRM Sudoku — Classification ({args.grid_size**2}x{args.grid_size**2})")
    print(f"  Device: {device} | Epochs: {args.epochs}")
    print(f"  Output: {num_cells} cells x {num_classes} classes = {output_dim} logits")
    print("=" * 65)

    train_ds = SudokuDataset(args.train_samples, args.grid_size, args.difficulty,
                             seed=args.seed, classification=True)
    test_ds  = SudokuDataset(args.test_samples,  args.grid_size, args.difficulty + 0.1,
                             seed=args.seed+1, classification=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model = HierarchicalReasoningModel(
        input_dim=num_cells,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_steps=args.num_steps,
        l_iterations=args.l_iterations,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20)

    best_puzzle_acc = 0
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        tr_loss, tr_cell, tr_puzzle = train_epoch(
            model, train_dl, optimizer, criterion, device, num_cells, num_classes)
        va_loss, va_cell, va_puzzle = evaluate(
            model, test_dl, criterion, device, num_cells, num_classes)

        scheduler.step(va_puzzle)

        if va_puzzle > best_puzzle_acc:
            best_puzzle_acc = va_puzzle

        epoch_bar.set_postfix(
            val_cell=f"{va_cell:.1f}%",
            val_puzzle=f"{va_puzzle:.1f}%",
            best=f"{best_puzzle_acc:.1f}%"
        )

        if epoch % 10 == 0:
            print(f"\n  Ep {epoch:>4} | Train cell={tr_cell:.1f}% puzzle={tr_puzzle:.1f}% "
                  f"| Val cell={va_cell:.1f}% puzzle={va_puzzle:.1f}%")

    print("\n" + "=" * 65)
    print(f"  Best puzzle accuracy : {best_puzzle_acc:.1f}%")
    print(f"  Final val cell acc   : {va_cell:.1f}%")
    print(f"  Paper claims         : ~100% puzzle accuracy")
    print("=" * 65)


if __name__ == "__main__":
    main()
