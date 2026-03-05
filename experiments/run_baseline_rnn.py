"""
Experiment: RNN Baseline vs HRM on Arithmetic

Trains a flat single-GRU baseline on the same arithmetic dataset
with identical hyperparameters, then compares against a saved HRM
checkpoint side by side.

Usage:
    python3 experiments/run_baseline_rnn.py
    python3 experiments/run_baseline_rnn.py --epochs 100
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import SyntheticReasoningDataset, get_dataloaders
from src.model import HierarchicalReasoningModel


# ─── Baseline Model ───────────────────────────────────────────────────────────

class RNNBaseline(nn.Module):
    """
    Flat single-timescale RNN baseline.
    Same hidden_dim as HRM for a fair parameter comparison.
    No hierarchy — one GRU, one pass.
    """

    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_projection(x)
        out, _ = self.gru(x)
        return self.output_head(out[:, -1, :])


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, train_loader, test_loader, epochs, lr, device, label):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device).unsqueeze(-1), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_loss, _ = evaluate(model, test_loader, criterion, device)
        train_loss = total_loss / len(train_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{epochs} "
                  f"| train: {train_loss:.4f} | val: {val_loss:.4f}")

    elapsed = time.time() - start
    model.load_state_dict(best_state)
    return best_val_loss, elapsed


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).unsqueeze(-1), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item() * x.size(0)
            total_mae += (pred - y).abs().sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_mae / n


# ─── Main ──────────────────────────────────────────────────────────────────────

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
    print(f"\nDevice: {device}")
    print(f"Epochs: {args.epochs} | Hidden: {args.hidden_dim} | LR: {args.lr}\n")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        train_size=8000,
        test_size=2000
    )
    criterion = nn.MSELoss()

    # ── Train RNN baseline ──────────────────────────────────────────────────
    print("Training RNN Baseline...")
    rnn = RNNBaseline(input_dim=1, hidden_dim=args.hidden_dim, output_dim=1).to(device)
    rnn_params = sum(p.numel() for p in rnn.parameters())

    rnn_best_loss, rnn_time = train(
        rnn, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr, device=device, label="RNN"
    )
    rnn_loss, rnn_mae = evaluate(rnn, test_loader, criterion, device)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'model_state_dict': rnn.state_dict()}, 'checkpoints/rnn_baseline.pt')
    print(f"  RNN checkpoint saved to checkpoints/rnn_baseline.pt\n")

    # ── Load or train HRM ───────────────────────────────────────────────────
    hrm = HierarchicalReasoningModel(
        input_dim=1, hidden_dim=args.hidden_dim, output_dim=1,
        num_steps=3, l_iterations=5, use_act=False
    ).to(device)
    hrm_params = sum(p.numel() for p in hrm.parameters())

    checkpoint_path = 'hrm_checkpoint.pt'
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        hrm.load_state_dict(ckpt['model_state_dict'], strict=False)
        print("HRM checkpoint loaded from hrm_checkpoint.pt")
        hrm_loss, hrm_mae = evaluate(hrm, test_loader, criterion, device)
        hrm_time = None
    else:
        print("No HRM checkpoint found — training HRM from scratch...")
        hrm_loss, hrm_time = train(
            hrm, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr, device=device, label="HRM"
        )
        hrm_loss, hrm_mae = evaluate(hrm, test_loader, criterion, device)

    # ── Comparison table ────────────────────────────────────────────────────
    print(f"\n{'━'*52}")
    print(f"  {'Model':<10} {'Params':>8} {'MSE Loss':>10} {'MAE':>8}")
    print(f"{'━'*52}")
    print(f"  {'RNN':<10} {rnn_params:>8,} {rnn_loss:>10.4f} {rnn_mae:>8.4f}")
    print(f"  {'HRM':<10} {hrm_params:>8,} {hrm_loss:>10.4f} {hrm_mae:>8.4f}")
    print(f"{'━'*52}")

    if hrm_loss < rnn_loss:
        improvement = ((rnn_loss - hrm_loss) / rnn_loss) * 100
        print(f"  HRM outperforms RNN by {improvement:.1f}% on MSE loss")
    else:
        gap = ((hrm_loss - rnn_loss) / rnn_loss) * 100
        print(f"  RNN outperforms HRM by {gap:.1f}% on MSE loss")

    print(f"{'━'*52}\n")


if __name__ == '__main__':
    main()
