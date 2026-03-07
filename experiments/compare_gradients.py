"""
Experiment: BPTT vs One-Step Gradient Approximation

Trains two identical HRM models:
  - BPTT        : gradients flow through all H-steps (standard backprop)
  - One-Step    : gradients detached between H-steps (paper's technique)

Compares:
  - Convergence speed (loss per epoch)
  - Final test loss
  - Training time

Usage:
    python3 experiments/compare_gradients.py
    python3 experiments/compare_gradients.py --epochs 100
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.model import HierarchicalReasoningModel
from src.dataset import get_dataloaders


def train_model(model, train_loader, test_loader, epochs, lr, device, label):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    start = time.time()

    epoch_bar = tqdm(range(epochs), desc=f"  [{label}]", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device).unsqueeze(-1)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}"
        )

    elapsed = time.time() - start
    return train_losses, val_losses, elapsed


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).unsqueeze(-1)
            y = y.to(device)
            pred = model(x)
            total += criterion(pred, y).item() * x.size(0)
    return total / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--hidden_dim', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--seed',       type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, train_size=8000, test_size=2000
    )

    print(f"\nDevice    : {device}")
    print(f"Epochs    : {args.epochs}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Seed      : {args.seed}\n")

    # ── BPTT model ────────────────────────────────────────────────────────────
    print("Training BPTT (standard backprop through all H-steps)...")
    bptt_model = HierarchicalReasoningModel(
        input_dim=1, hidden_dim=args.hidden_dim, output_dim=1,
        num_steps=3, l_iterations=5, one_step_grad=False
    ).to(device)

    bptt_train, bptt_val, bptt_time = train_model(
        bptt_model, train_loader, test_loader,
        args.epochs, args.lr, device, "BPTT"
    )

    # ── One-step model ────────────────────────────────────────────────────────
    print("\nTraining One-Step (detached gradients between H-steps)...")
    onestep_model = HierarchicalReasoningModel(
        input_dim=1, hidden_dim=args.hidden_dim, output_dim=1,
        num_steps=3, l_iterations=5, one_step_grad=True
    ).to(device)

    onestep_train, onestep_val, onestep_time = train_model(
        onestep_model, train_loader, test_loader,
        args.epochs, args.lr, device, "One-Step"
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    best_bptt     = min(bptt_val)
    best_onestep  = min(onestep_val)
    bptt_ep       = bptt_val.index(best_bptt) + 1
    onestep_ep    = onestep_val.index(best_onestep) + 1

    # Convergence: epoch where val loss first drops below 2x final best
    threshold = best_bptt * 2
    bptt_conv    = next((i+1 for i, v in enumerate(bptt_val)    if v < threshold), args.epochs)
    onestep_conv = next((i+1 for i, v in enumerate(onestep_val) if v < threshold), args.epochs)

    print(f"\n{'━'*62}")
    print(f"  {'Method':<16} {'Best MSE':>10} {'Best Epoch':>11} {'Conv. Epoch':>12} {'Time':>8}")
    print(f"{'━'*62}")
    print(f"  {'BPTT':<16} {best_bptt:>10.4f} {bptt_ep:>11d} {bptt_conv:>12d} {bptt_time:>7.1f}s")
    print(f"  {'One-Step':<16} {best_onestep:>10.4f} {onestep_ep:>11d} {onestep_conv:>12d} {onestep_time:>7.1f}s")
    print(f"{'━'*62}")

    speedup = bptt_time / onestep_time
    if speedup > 1:
        print(f"  One-Step is {speedup:.2f}x faster to train")
    else:
        print(f"  BPTT is {1/speedup:.2f}x faster to train")

    if best_onestep <= best_bptt * 1.05:
        print(f"  One-Step matches BPTT quality (within 5%)")
    else:
        gap = ((best_onestep - best_bptt) / best_bptt) * 100
        print(f"  BPTT achieves {gap:.1f}% better MSE (more training epochs may close gap)")

    print(f"{'━'*62}\n")

    # Save both checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'model_state_dict': bptt_model.state_dict()},
               'checkpoints/hrm_bptt.pt')
    torch.save({'model_state_dict': onestep_model.state_dict()},
               'checkpoints/hrm_onestep.pt')
    print("Checkpoints saved: checkpoints/hrm_bptt.pt, checkpoints/hrm_onestep.pt")


if __name__ == '__main__':
    main()
