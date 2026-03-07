"""
Experiment: Training the ACT (Adaptive Computation Time) Module

Trains the Q-learning halting module to decide when to stop reasoning.
Reward signal:
    +1.0  correct prediction and halted
    -1.0  wrong prediction and halted
    -0.1  per step penalty (encourages early halting)

Compares fixed depth (3 H-steps always) vs ACT adaptive depth.

Usage:
    python3 experiments/train_act.py
    python3 experiments/train_act.py --epochs 100 --threshold 0.05
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import HierarchicalReasoningModel
from src.dataset import SyntheticReasoningDataset, get_dataloaders


# ─── Rewards ──────────────────────────────────────────────────────────────────

REWARD_CORRECT_HALT  =  1.0   # halted and got it right
REWARD_WRONG_HALT    = -1.0   # halted and got it wrong
REWARD_STEP_PENALTY  = -0.1   # paid per step to encourage early halting


def compute_reward(pred, target, step, max_steps, threshold=0.05):
    """
    Compute per-sample Q-learning reward.

    Args:
        pred    : [batch, 1] model prediction
        target  : [batch, 1] ground truth
        step    : current H-step index (0-based)
        max_steps: total allowed H-steps
        threshold: MAE below this = correct

    Returns:
        reward: [batch] float tensor
    """
    mae = (pred - target).abs().squeeze(-1)          # [batch]
    correct = (mae < threshold).float()              # 1 if correct, 0 if wrong
    step_cost = REWARD_STEP_PENALTY * (step + 1)

    reward = torch.where(
        correct.bool(),
        torch.full_like(correct, REWARD_CORRECT_HALT + step_cost),
        torch.full_like(correct, REWARD_WRONG_HALT   + step_cost)
    )
    return reward


# ─── ACT Training Loop ────────────────────────────────────────────────────────

def train_act(model, train_loader, test_loader, args, device):
    """
    Two-phase training:
    Phase 1 — freeze H/L modules, train ACT Q-network only
    Phase 2 — evaluate adaptive vs fixed depth
    """
    criterion = nn.MSELoss()

    # Freeze H-module, L-module, output head — only train ACT
    for name, param in model.named_parameters():
        if 'act_module' not in name:
            param.requires_grad = False

    act_optimizer = optim.Adam(model.act_module.parameters(), lr=args.lr)

    print(f"\n{'='*60}")
    print(f"  ACT Q-Network Training")
    print(f"  Trainable params : {sum(p.numel() for p in model.act_module.parameters()):,}")
    print(f"  Frozen params    : {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"  Reward threshold : MAE < {args.threshold}")
    print(f"  Epochs           : {args.epochs}")
    print(f"{'='*60}\n")

    best_avg_steps = model.num_steps  # baseline: always uses max steps
    act_losses = []

    epoch_bar = tqdm(range(args.epochs), desc="Training ACT", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        epoch_q_loss = 0.0
        total_steps_used = 0
        total_samples = 0

        batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch+1:3d}/{args.epochs}",
                         leave=False, unit="batch")

        for x, y in batch_bar:
            x = x.to(device).unsqueeze(-1)   # [B, 3, 1]
            y = y.to(device)                  # [B, 1]

            batch_size = x.size(0)
            h_state = None
            current_input = x
            q_loss_batch = torch.tensor(0.0, device=device)
            steps_used = torch.ones(batch_size, device=device) * model.num_steps

            # Run through H-steps, collect Q-learning loss at each step
            for h_step in range(model.num_steps):
                with torch.no_grad():
                    control_signal, h_state = model.high_level(current_input, h_state)
                    l_state, _ = model.low_level(control_signal)
                    pred = model.output_head(l_state)

                # ACT decision at this step
                action, q_values = model.act_module(l_state.detach())

                # Compute reward for halting here
                reward = compute_reward(pred.detach(), y, h_step,
                                        model.num_steps, args.threshold)

                # Q-learning target: reward if halt, else bootstrap
                q_target = q_values.clone().detach()
                q_target[:, 0] = reward                         # Q(halt) = reward
                q_target[:, 1] = reward + 0.9 * q_values[:, 0].detach()  # Q(continue)

                # MSE loss on Q-values
                q_loss_batch = q_loss_batch + nn.functional.mse_loss(q_values, q_target)

                # Track which samples halted
                halted = (action == 0)
                for i in range(batch_size):
                    if halted[i] and steps_used[i] == model.num_steps:
                        steps_used[i] = h_step + 1

                # Update input for next step
                current_input = l_state.detach().unsqueeze(1)

            act_optimizer.zero_grad()
            q_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.act_module.parameters(), 1.0)
            act_optimizer.step()

            epoch_q_loss += q_loss_batch.item()
            total_steps_used += steps_used.sum().item()
            total_samples += batch_size

            avg_steps = total_steps_used / total_samples
            batch_bar.set_postfix(q_loss=f"{q_loss_batch.item():.4f}",
                                  avg_steps=f"{avg_steps:.2f}")

        avg_q_loss = epoch_q_loss / len(train_loader)
        avg_steps_epoch = total_steps_used / total_samples
        act_losses.append(avg_q_loss)

        if avg_steps_epoch < best_avg_steps:
            best_avg_steps = avg_steps_epoch
            torch.save({'act_state_dict': model.act_module.state_dict()},
                       'checkpoints/act_module.pt')

        epoch_bar.set_postfix(q_loss=f"{avg_q_loss:.4f}",
                              avg_steps=f"{avg_steps_epoch:.2f}/{model.num_steps}")

    return act_losses


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_comparison(model, test_loader, device, threshold):
    """Compare fixed depth vs ACT adaptive depth on test set."""
    criterion = nn.MSELoss()
    model.eval()

    fixed_loss = 0.0
    act_loss = 0.0
    act_steps_total = 0
    n = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating", leave=False):
            x = x.to(device).unsqueeze(-1)
            y = y.to(device)
            batch_size = x.size(0)

            # Fixed: always use all H-steps (use_act=False)
            model.use_act = False
            pred_fixed = model(x)
            fixed_loss += criterion(pred_fixed, y).item() * batch_size

            # ACT: adaptive halting
            model.use_act = True
            h_state = None
            current_input = x
            final_pred = None
            steps_used = batch_size * model.num_steps  # default

            for h_step in range(model.num_steps):
                control_signal, h_state = model.high_level(current_input, h_state)
                l_state, _ = model.low_level(control_signal)
                pred = model.output_head(l_state)
                action, _ = model.act_module(l_state)

                if action.sum() == 0 or h_step == model.num_steps - 1:
                    final_pred = pred
                    steps_used = (h_step + 1) * batch_size
                    break

                current_input = l_state.unsqueeze(1)

            if final_pred is None:
                final_pred = pred

            act_loss += criterion(final_pred, y).item() * batch_size
            act_steps_total += steps_used
            n += batch_size

    return {
        'fixed_mse':   fixed_loss / n,
        'act_mse':     act_loss / n,
        'fixed_steps': model.num_steps,
        'act_steps':   act_steps_total / n,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--threshold',  type=float, default=0.05,
                        help='MAE threshold to count prediction as correct')
    parser.add_argument('--hidden_dim', type=int,   default=32)
    parser.add_argument('--num_steps',  type=int,   default=3)
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
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, train_size=8000, test_size=2000
    )

    # Load HRM with ACT enabled
    model = HierarchicalReasoningModel(
        input_dim=1, hidden_dim=args.hidden_dim, output_dim=1,
        num_steps=args.num_steps, l_iterations=5, use_act=True, min_h_steps=1
    ).to(device)

    # Load pretrained HRM weights if available
    ckpt_path = 'hrm_checkpoint.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded HRM weights from {ckpt_path}")
    else:
        print("No HRM checkpoint found — using random weights")

    os.makedirs('checkpoints', exist_ok=True)

    # Train ACT
    train_act(model, train_loader, test_loader, args, device)

    # Compare fixed vs adaptive
    print("\nRunning final comparison...")
    results = evaluate_comparison(model, test_loader, device, args.threshold)

    print(f"\n{'━'*52}")
    print(f"  {'Mode':<16} {'MSE Loss':>10} {'Avg H-steps':>12}")
    print(f"{'━'*52}")
    print(f"  {'Fixed (always 3)':<16} {results['fixed_mse']:>10.4f} {results['fixed_steps']:>12.1f}")
    print(f"  {'ACT (adaptive)':<16} {results['act_mse']:>10.4f} {results['act_steps']:>12.2f}")
    print(f"{'━'*52}")

    step_reduction = (1 - results['act_steps'] / results['fixed_steps']) * 100
    if step_reduction > 0:
        print(f"  ACT uses {step_reduction:.1f}% fewer steps on average")
    else:
        print(f"  ACT needs more training to reduce steps (expected for early runs)")

    print(f"{'━'*52}\n")
    print(f"ACT checkpoint saved to checkpoints/act_module.pt")


if __name__ == '__main__':
    main()
