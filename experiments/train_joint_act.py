"""
Experiment: Joint ACT + HRM Training
Both HRM and ACT train together — HRM learns to produce good early outputs,
ACT learns when early is good enough. Fixes the frozen-HRM accuracy drop.

Key difference from train_act.py:
- HRM weights are NOT frozen
- Joint loss = task_loss (MSE) + lambda * Q_loss (ACT)
- Both update simultaneously each batch

Usage:
    python3 experiments/train_joint_act.py
    python3 experiments/train_joint_act.py --epochs 100 --lam 0.1
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import HierarchicalReasoningModel
from src.dataset import get_dataloaders
from src.utils import set_seed

REWARD_CORRECT_HALT = 1.0
REWARD_WRONG_HALT   = -1.0
REWARD_STEP_PENALTY = -0.1


def compute_reward(pred, target, step, threshold=0.05):
    mae = (pred - target).abs().squeeze(-1)
    correct = (mae < threshold).float()
    step_cost = REWARD_STEP_PENALTY * (step + 1)
    return torch.where(
        correct.bool(),
        torch.full_like(correct, REWARD_CORRECT_HALT + step_cost),
        torch.full_like(correct, REWARD_WRONG_HALT   + step_cost)
    )


def train_epoch(model, loader, optimizer, criterion, device, lam, threshold):
    model.train()
    total_task_loss, total_q_loss, total_steps, total_n = 0, 0, 0, 0

    pbar = tqdm(loader, leave=False, desc="  batches")
    for x, y in pbar:
        x = x.to(device).unsqueeze(-1)
        y = y.to(device)
        B = x.size(0)

        h_state = None
        current_input = x
        q_loss_batch = torch.tensor(0.0, device=device)
        steps_used = torch.ones(B, device=device) * model.num_steps
        final_pred = None

        for h_step in range(model.num_steps):
            control_signal, h_state = model.high_level(current_input, h_state)
            l_state, _ = model.low_level(control_signal)
            pred = model.output_head(l_state)

            # ACT decision
            action, q_values = model.act_module(l_state)

            # Q-learning reward
            reward = compute_reward(pred.detach(), y, h_step, threshold)
            q_target = q_values.clone().detach()
            q_target[:, 0] = reward
            q_target[:, 1] = reward + 0.9 * q_values[:, 0].detach()
            q_loss_batch = q_loss_batch + nn.functional.mse_loss(q_values, q_target)

            # Track halt
            halted = (action == 0)
            for i in range(B):
                if halted[i] and steps_used[i] == model.num_steps:
                    steps_used[i] = h_step + 1

            if halted.all() or h_step == model.num_steps - 1:
                final_pred = pred
                break

            current_input = l_state.detach().unsqueeze(1)

        if final_pred is None:
            final_pred = pred

        # Joint loss: task + ACT — both HRM and ACT update
        task_loss = criterion(final_pred, y)
        joint_loss = task_loss + lam * q_loss_batch

        optimizer.zero_grad()
        joint_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        avg_steps = steps_used.mean().item()
        total_task_loss += task_loss.item()
        total_q_loss += q_loss_batch.item()
        total_steps += steps_used.sum().item()
        total_n += B
        pbar.set_postfix(task=f"{task_loss.item():.4f}",
                         q=f"{q_loss_batch.item():.4f}",
                         steps=f"{avg_steps:.1f}")

    n = len(loader)
    return total_task_loss / n, total_q_loss / n, total_steps / total_n


def evaluate(model, loader, criterion, device):
    model.eval()
    fixed_loss, act_loss, act_steps, n = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).unsqueeze(-1)
            y = y.to(device)
            B = x.size(0)

            # Fixed: all steps
            out = model(x)
            fixed_loss += criterion(out, y).item()

            # ACT: adaptive
            h_state, current_input, final_pred = None, x, None
            for h_step in range(model.num_steps):
                ctrl, h_state = model.high_level(current_input, h_state)
                l_state, _ = model.low_level(ctrl)
                pred = model.output_head(l_state)
                action, _ = model.act_module(l_state)
                if action.sum() == 0 or h_step == model.num_steps - 1:
                    final_pred = pred
                    act_steps += (h_step + 1) * B
                    break
                current_input = l_state.unsqueeze(1)
            if final_pred is None:
                final_pred = pred
            act_loss += criterion(final_pred, y).item()
            n += B

    nb = len(loader)
    return fixed_loss / nb, act_loss / nb, act_steps / n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lam',        type=float, default=0.1,
                   help='Weight of ACT Q-loss in joint loss')
    p.add_argument('--threshold',  type=float, default=0.05)
    p.add_argument('--hidden_dim', type=int,   default=32)
    p.add_argument('--num_steps',  type=int,   default=3)
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, train_size=8000, test_size=2000)

    model = HierarchicalReasoningModel(
        input_dim=1, hidden_dim=args.hidden_dim, output_dim=1,
        num_steps=args.num_steps, l_iterations=5,
        use_act=True, min_h_steps=1
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print("=" * 60)
    print(f"  Joint ACT + HRM Training")
    print(f"  Device: {device} | Params: {params:,}")
    print(f"  λ (ACT weight): {args.lam} | Epochs: {args.epochs}")
    print(f"  Both HRM + ACT weights update each batch")
    print("=" * 60)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"\n  {'Ep':>4}  {'TaskLoss':>9}  {'QLoss':>7}  {'AvgSteps':>9}  {'FixedMSE':>9}  {'ACTMSE':>7}")
    print("  " + "-" * 55)

    best_act_mse = float('inf')
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        task_loss, q_loss, avg_steps = train_epoch(
            model, train_loader, optimizer, criterion, device, args.lam, args.threshold)
        fixed_mse, act_mse, val_steps = evaluate(model, test_loader, criterion, device)

        if act_mse < best_act_mse:
            best_act_mse = act_mse
            torch.save({'model_state_dict': model.state_dict()},
                       'checkpoints/joint_act_hrm.pt')

        epoch_bar.set_postfix(task=f"{task_loss:.4f}", act_mse=f"{act_mse:.4f}",
                              steps=f"{val_steps:.1f}")

        if epoch % 10 == 0:
            print(f"  {epoch:>4}  {task_loss:>9.4f}  {q_loss:>7.4f}  {avg_steps:>9.2f}  "
                  f"{fixed_mse:>9.4f}  {act_mse:>7.4f}")

    print("\n" + "=" * 60)
    step_reduction = (1 - val_steps / args.num_steps) * 100
    print(f"  Best ACT MSE    : {best_act_mse:.4f}")
    print(f"  Avg steps used  : {val_steps:.2f} / {args.num_steps}")
    print(f"  Step reduction  : {step_reduction:.1f}%")
    print(f"  vs frozen-HRM   : ACT trained jointly — HRM co-adapted")
    print("=" * 60)


if __name__ == '__main__':
    main()
