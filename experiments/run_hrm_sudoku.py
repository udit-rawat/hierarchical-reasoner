"""
Experiment: Training HRM on Sudoku Puzzles

This script validates the paper's claim of near-perfect Sudoku solving.
Paper configuration:
- 1000 training samples
- 27M parameters (approx)
- Near-perfect accuracy on 9×9 Sudoku

Usage:
    python3 experiments/run_hrm_sudoku.py
    python3 experiments/run_hrm_sudoku.py --quick_test  # Fast 4×4 test
"""
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils import set_seed, save_checkpoint, load_checkpoint, log_metrics
from src.datasetSudoku import SudokuDataset
from src.model import HierarchicalReasoningModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train HRM on Sudoku')

    # Dataset args
    parser.add_argument('--grid_size', type=int, default=3,
                        help='Grid size (3=9×9, 2=4×4)')
    parser.add_argument('--train_samples', type=int,
                        default=1000, help='Training samples')
    parser.add_argument('--test_samples', type=int,
                        default=200, help='Test samples')
    parser.add_argument('--difficulty', type=float,
                        default=0.5, help='Puzzle difficulty (0-1)')

    # Model args
    parser.add_argument('--hidden_dim', type=int,
                        default=128, help='Hidden dimension')
    parser.add_argument('--num_steps', type=int,
                        default=5, help='H-module steps')
    parser.add_argument('--l_iterations', type=int, default=10,
                        help='L-module iterations per H-step')
    parser.add_argument('--use_act', action='store_true',
                        help='Use adaptive computation time')

    # Training args
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_scheduler',
                        action='store_true', help='Use LR scheduler')
    parser.add_argument('--grad_clip', type=float,
                        default=1.0, help='Gradient clipping value')

    # Experiment args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test mode (4×4, 100 samples)')
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str,
                        default='metal', help='Device (metal/cpu)')

    return parser.parse_args()


def calculate_accuracy(predictions, targets, threshold=0.1):
    """
    Calculate Sudoku solving accuracy.

    Args:
        predictions: Model outputs [batch, 81]
        targets: Ground truth [batch, 81]
        threshold: Tolerance for matching (after rounding)

    Returns:
        cell_accuracy: % of cells correct
        puzzle_accuracy: % of fully solved puzzles
    """
    # Denormalize (predictions are in [0,1], multiply by 9 for 9×9)
    grid_size = int(predictions.shape[1] ** 0.5)
    predictions = predictions * grid_size
    targets = targets * grid_size

    # Round to nearest integer
    pred_rounded = torch.round(predictions)
    targets_rounded = torch.round(targets)

    # Cell-level accuracy
    correct_cells = (torch.abs(pred_rounded - targets_rounded)
                     < threshold).float()
    cell_accuracy = correct_cells.mean().item() * 100

    # Puzzle-level accuracy (all cells must be correct)
    correct_puzzles = (correct_cells.mean(dim=1) == 1.0).float()
    puzzle_accuracy = correct_puzzles.mean().item() * 100

    return cell_accuracy, puzzle_accuracy


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_cell_acc = 0
    total_puzzle_acc = 0

    pbar = tqdm(dataloader, leave=False, desc="  batches")
    for batch_idx, (puzzles, solutions) in enumerate(pbar):
        puzzles = puzzles.to(device)
        solutions = solutions.to(device)

        optimizer.zero_grad()

        # --- Safe forward pass (handles multiple return values) ---
        out = model(puzzles)
        if isinstance(out, (tuple, list)):
            outputs = out[0]  # first element = predictions
        else:
            outputs = out
        # ----------------------------------------------------------

        loss = criterion(outputs, solutions)
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        cell_acc, puzzle_acc = calculate_accuracy(outputs.detach(), solutions)
        total_cell_acc += cell_acc
        total_puzzle_acc += puzzle_acc
        pbar.set_postfix(loss=f"{loss.item():.4f}", cell=f"{cell_acc:.1f}%")

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'cell_accuracy': total_cell_acc / n_batches,
        'puzzle_accuracy': total_puzzle_acc / n_batches
    }


def evaluate(model, dataloader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    total_cell_acc = 0
    total_puzzle_acc = 0

    with torch.no_grad():
        for puzzles, solutions in dataloader:
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)

            # --- Safe forward pass (handles multiple return values) ---
            out = model(puzzles)
            if isinstance(out, (tuple, list)):
                outputs = out[0]
            else:
                outputs = out

            loss = criterion(outputs, solutions)

            # Metrics
            total_loss += loss.item()
            cell_acc, puzzle_acc = calculate_accuracy(outputs, solutions)
            total_cell_acc += cell_acc
            total_puzzle_acc += puzzle_acc

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'cell_accuracy': total_cell_acc / n_batches,
        'puzzle_accuracy': total_puzzle_acc / n_batches
    }


def visualize_prediction(model, dataset, device, save_path='sudoku_prediction.png'):
    """Visualize a sample prediction"""
    model.eval()

    # Get a sample
    puzzle, solution = dataset[0]
    puzzle = puzzle.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(puzzle)
        # Handle both single return and tuple return
        if isinstance(out, (tuple, list)):
            prediction = out[0]
            trajectory = out[1] if len(out) > 1 else None
        else:
            prediction = out
            trajectory = None

    # Denormalize
    grid_size = int(puzzle.shape[1] ** 0.5)
    puzzle = puzzle.squeeze().cpu() * grid_size
    solution = solution * grid_size
    prediction = prediction.squeeze().cpu() * grid_size

    # Round predictions
    pred_rounded = torch.round(prediction)

    # Reshape to 2D grids
    size = int(len(puzzle) ** 0.5)
    puzzle_grid = puzzle.reshape(size, size).numpy()
    solution_grid = solution.reshape(size, size).numpy()
    pred_grid = pred_rounded.reshape(size, size).numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Puzzle
    axes[0].imshow(puzzle_grid, cmap='Blues', vmin=0, vmax=grid_size)
    axes[0].set_title('Input Puzzle\n(0 = empty)',
                      fontsize=14, fontweight='bold')
    for i in range(size):
        for j in range(size):
            val = puzzle_grid[i, j]
            axes[0].text(j, i, f'{int(val)}' if val > 0 else '.',
                         ha='center', va='center', fontsize=10)

    # Prediction
    axes[1].imshow(pred_grid, cmap='Greens', vmin=0, vmax=grid_size)
    axes[1].set_title('HRM Prediction', fontsize=14, fontweight='bold')
    for i in range(size):
        for j in range(size):
            val = pred_grid[i, j]
            axes[1].text(j, i, f'{int(val)}', ha='center',
                         va='center', fontsize=10)

    # Ground truth
    axes[2].imshow(solution_grid, cmap='Oranges', vmin=0, vmax=grid_size)
    axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
    for i in range(size):
        for j in range(size):
            val = solution_grid[i, j]
            axes[2].text(j, i, f'{int(val)}', ha='center',
                         va='center', fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")


def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'],
                 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-',
                 label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cell accuracy
    axes[1].plot(epochs, history['train_cell_acc'],
                 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_cell_acc'],
                 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Cell Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Puzzle accuracy
    axes[2].plot(epochs, history['train_puzzle_acc'],
                 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_puzzle_acc'],
                 'r-', label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_title('Puzzle Accuracy (Fully Solved)',
                      fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training curves saved to {save_path}")


def main():
    args = parse_args()

    # Quick test mode
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE (4×4 Sudoku, 100 samples)")
        args.grid_size = 2
        args.train_samples = 100
        args.test_samples = 20
        args.epochs = 20
        args.hidden_dim = 64

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"🔬 EXPERIMENT: HRM on Sudoku")
    print(f"Device: {device}")

    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = SudokuDataset(
        num_samples=args.train_samples,
        grid_size=args.grid_size,
        difficulty=args.difficulty,
        seed=args.seed
    )

    test_dataset = SudokuDataset(
        num_samples=args.test_samples,
        grid_size=args.grid_size,
        difficulty=args.difficulty + 0.1,  # Slightly harder test set
        seed=args.seed + 1
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    input_dim = (args.grid_size ** 2) ** 2  # 81 for 9×9
    print(f"\n🧠 Creating HRM model...")
    print(f"  Input/Output dim: {input_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  H-steps: {args.num_steps}")
    print(f"  L-iterations: {args.l_iterations}")
    print(f"  Total computation depth: {args.num_steps * args.l_iterations}")

    model = HierarchicalReasoningModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=input_dim,
        num_steps=args.num_steps,
        l_iterations=args.l_iterations,
        use_act=args.use_act
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cell_acc': [], 'val_cell_acc': [],
        'train_puzzle_acc': [], 'val_puzzle_acc': []
    }

    best_puzzle_acc = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    print(f"\n🚀 TRAINING START")
    start_time = time.time()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip)

        # Validate
        val_metrics = evaluate(model, test_loader, criterion, device)

        # Scheduler step
        if scheduler:
            scheduler.step(val_metrics['loss'])

        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_cell_acc'].append(train_metrics['cell_accuracy'])
        history['val_cell_acc'].append(val_metrics['cell_accuracy'])
        history['train_puzzle_acc'].append(train_metrics['puzzle_accuracy'])
        history['val_puzzle_acc'].append(val_metrics['puzzle_accuracy'])

        # Print progress
        epoch_time = time.time() - epoch_start
        epoch_bar.set_postfix(
            loss=f"{val_metrics['loss']:.4f}",
            cell=f"{val_metrics['cell_accuracy']:.1f}%",
            puzzle=f"{val_metrics['puzzle_accuracy']:.1f}%",
            t=f"{epoch_time:.0f}s"
        )
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Cell: {train_metrics['cell_accuracy']:.2f}% | "
              f"Puzzle: {train_metrics['puzzle_accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Cell: {val_metrics['cell_accuracy']:.2f}% | "
              f"Puzzle: {val_metrics['puzzle_accuracy']:.2f}%")

        # Save best model
        if val_metrics['puzzle_accuracy'] > best_puzzle_acc:
            best_puzzle_acc = val_metrics['puzzle_accuracy']
            checkpoint_path = os.path.join(args.save_dir, 'hrm_sudoku_best.pt')
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'],
                checkpoint_path,
                extra_info={
                    'cell_accuracy': val_metrics['cell_accuracy'],
                    'puzzle_accuracy': val_metrics['puzzle_accuracy'],
                    'args': vars(args)
                }
            )
            print(f"  ✅ Best model saved (Puzzle Acc: {best_puzzle_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation puzzle accuracy: {best_puzzle_acc:.2f}%")

    # Plot results
    print("\n📊 Generating plots...")
    plot_training_curves(history, os.path.join(
        args.save_dir, 'training_curves.png'))
    visualize_prediction(model, test_dataset, device, os.path.join(
        args.save_dir, 'sample_prediction.png'))

    # Final evaluation
    print("\n📈 Final Evaluation on Test Set:")
    final_metrics = evaluate(model, test_loader, criterion, device)
    log_metrics(final_metrics)

    print(f"\n{'='*70}")
    print(f"🎉 EXPERIMENT COMPLETE")
    print(f"Checkpoints saved to: {args.save_dir}/")
    print(f"Best puzzle accuracy: {best_puzzle_acc:.2f}%")

    # Compare with paper
    print(f"\n📄 Paper Comparison:")
    print(f"  Paper claims: ~100% accuracy on Sudoku")
    print(f"  Your result: {best_puzzle_acc:.2f}% puzzle accuracy")
    if best_puzzle_acc > 90:
        print(f"  ✅ SUCCESS! Near-perfect performance achieved!")
    elif best_puzzle_acc > 70:
        print(f"  ⚠️ Good but needs tuning (try more epochs/larger model)")
    else:
        print(f"  ❌ Needs improvement (check hyperparameters)")


if __name__ == "__main__":
    main()
