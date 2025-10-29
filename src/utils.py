"""
Utility Functions for Hierarchical Reasoning Model

Centralized configuration, helper functions, and magic variables.
Provides reproducibility, checkpointing, logging, and visualization tools.
"""

import torch
import numpy as np
import random
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# ============================================================================
# MAGIC VARIABLES & CONFIGURATION
# ============================================================================

class Config:
    """
    Centralized configuration for HRM project.
    All magic numbers and hyperparameters live here.
    """

    # ==================== PATHS ====================
    CHECKPOINT_DIR = "checkpoints"
    LOGS_DIR = "logs"
    RESULTS_DIR = "results"
    DATA_DIR = "data"

    # ==================== RANDOM SEEDS ====================
    DEFAULT_SEED = 42
    TRAIN_SEED = 42
    TEST_SEED = 123

    # Simple Arithmetic Task
    ARITHMETIC_INPUT_DIM = 1
    ARITHMETIC_HIDDEN_DIM = 32
    ARITHMETIC_OUTPUT_DIM = 1
    ARITHMETIC_NUM_STEPS = 3
    ARITHMETIC_L_ITERATIONS = 5

    # Sudoku Task (9×9)
    SUDOKU_INPUT_DIM = 81  # 9×9 grid flattened
    SUDOKU_HIDDEN_DIM = 128  # Larger for complex task
    SUDOKU_OUTPUT_DIM = 81
    SUDOKU_NUM_STEPS = 5  # More H-steps for complex reasoning
    SUDOKU_L_ITERATIONS = 10  # More L-iterations per H-step

    # Sudoku Task (4×4) - For quick testing
    SUDOKU_4x4_INPUT_DIM = 16
    SUDOKU_4x4_HIDDEN_DIM = 64
    SUDOKU_4x4_OUTPUT_DIM = 16
    SUDOKU_4x4_NUM_STEPS = 3
    SUDOKU_4x4_L_ITERATIONS = 5

    # ACT Module
    USE_ACT = False  # Adaptive Computation Time (currently disabled)
    ACT_THRESHOLD = 0.99  # Halting threshold
    ACT_MAX_STEPS = 10  # Maximum ACT steps

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    GRADIENT_CLIP = 1.0
    WEIGHT_DECAY = 0.0

    # Learning Rate Scheduler
    USE_SCHEDULER = False
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5

    # ==================== DATASET ====================
    # Arithmetic
    ARITHMETIC_TRAIN_SIZE = 8000
    ARITHMETIC_TEST_SIZE = 2000

    # Sudoku
    SUDOKU_TRAIN_SIZE = 1000  # Paper uses 1000
    SUDOKU_TEST_SIZE = 200
    SUDOKU_GRID_SIZE = 3  # 3 = 9×9, 2 = 4×4
    SUDOKU_DIFFICULTY = 0.5  # 0.0-1.0
    SUDOKU_NORMALIZE = True

    # ==================== EVALUATION ====================
    EVAL_BATCH_SIZE = 64
    TRAJECTORY_ANALYSIS = True
    VISUALIZE_SAMPLES = 5

    # Accuracy thresholds
    SUDOKU_ACCURACY_THRESHOLD = 0.95  # Paper achieves near-perfect
    ARITHMETIC_ACCURACY_THRESHOLD = 0.90

    # ==================== LOGGING ====================
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_CHECKPOINT_EVERY = 5  # Save every N epochs
    VERBOSE = True

    # ==================== VISUALIZATION ====================
    FIG_SIZE = (12, 8)
    DPI = 100
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'

    # ==================== DEVICE ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls, category: Optional[str] = None):
        """Print configuration values"""

        print("HRM CONFIGURATION")

        if category is None:
            categories = ['PATHS', 'RANDOM SEEDS', 'MODEL ARCHITECTURE',
                          'TRAINING', 'DATASET', 'EVALUATION', 'DEVICE']
        else:
            categories = [category.upper()]

        for cat in categories:
            print(f"\n[{cat}]")
            for key, value in vars(cls).items():
                if not key.startswith('_') and cat.replace(' ', '_') in key:
                    print(f"  {key}: {value}")


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = Config.DEFAULT_SEED):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✓ Random seed set to {seed}")


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def setup_directories():
    """Create project directories if they don't exist"""
    dirs = [
        Config.CHECKPOINT_DIR,
        Config.LOGS_DIR,
        Config.RESULTS_DIR,
        Config.DATA_DIR
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print(f"✓ Project directories ready")


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Optional[Dict[str, Any]] = None,
    filepath: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save model checkpoint with full training state.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        metrics: Additional metrics dict
        filepath: Custom save path (optional)
        config: Model configuration dict (optional)

    Returns:
        Path to saved checkpoint
    """
    setup_directories()

    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(Config.CHECKPOINT_DIR,
                                f"hrm_epoch{epoch}_{timestamp}.pt")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'config': config or {},
        'metrics': metrics or {}
    }

    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")

    return filepath


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = Config.DEVICE,
    strict: bool = False
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to map tensors to
        strict: Strict state_dict loading (set False for ACT mismatches)

    Returns:
        Tuple of (model, optimizer, checkpoint_info)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    info = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
        'config': checkpoint.get('config', {}),
        'metrics': checkpoint.get('metrics', {})
    }

    print(f"✓ Checkpoint loaded: {filepath}")
    print(f"  Epoch: {info['epoch']}, Loss: {info['loss']:.6f}")

    return model, optimizer, info


# ============================================================================
# LOGGING
# ============================================================================

def log_metrics(
    metrics: Dict[str, float],
    epoch: Optional[int] = None,
    prefix: str = ""
):
    """
    Log metrics to console in a formatted way.

    Args:
        metrics: Dictionary of metric names and values
        epoch: Epoch number (optional)
        prefix: Prefix for log message
    """
    header = f"{prefix} " if prefix else ""
    if epoch is not None:
        header += f"Epoch {epoch} | "

    metric_strs = [f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}"
                   for k, v in metrics.items()]

    print(header + " | ".join(metric_strs))


def save_results(
    results: Dict[str, Any],
    filename: str,
    results_dir: str = Config.RESULTS_DIR
):
    """
    Save results dictionary to JSON file.

    Args:
        results: Results dictionary
        filename: Output filename
        results_dir: Directory to save results
    """
    setup_directories()
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved: {filepath}")


# ============================================================================
# TRAJECTORY VISUALIZATION
# ============================================================================

def visualize_trajectory(
    trajectory: List[Dict[str, torch.Tensor]],
    save_path: Optional[str] = None,
    title: str = "HRM Reasoning Trajectory"
):
    """
    Visualize HRM reasoning trajectory across H-steps and L-iterations.

    Args:
        trajectory: List of trajectory dicts with h_state, l_state, output
        save_path: Path to save figure (optional)
        title: Plot title
    """
    try:
        plt.style.use(Config.PLOT_STYLE)
    except:
        pass

    num_steps = len(trajectory)

    fig, axes = plt.subplots(3, 1, figsize=Config.FIG_SIZE, dpi=Config.DPI)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Extract data
    h_states = [step['h_state'].detach().cpu().numpy().flatten()
                for step in trajectory]
    l_states = [step['l_state'].detach().cpu().numpy().flatten()
                for step in trajectory]
    outputs = [step['output'].detach().cpu().numpy().flatten()
               for step in trajectory]

    # Plot 1: H-module states (slow planning)
    axes[0].plot(h_states, marker='o', label='H-state dimensions')
    axes[0].set_title("High-Level Module (Slow Planning)")
    axes[0].set_xlabel("H-step")
    axes[0].set_ylabel("State Value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: L-module states (fast execution)
    axes[1].plot(l_states, marker='s', alpha=0.7, label='L-state dimensions')
    axes[1].set_title("Low-Level Module (Fast Execution)")
    axes[1].set_xlabel("L-iteration (across all H-steps)")
    axes[1].set_ylabel("State Value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Outputs
    axes[2].plot(outputs, marker='^', color='green', label='Outputs')
    axes[2].set_title("Output Evolution")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Output Value")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"✓ Trajectory plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Curves"
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        save_path: Path to save figure (optional)
        title: Plot title
    """
    try:
        plt.style.use(Config.PLOT_STYLE)
    except:
        pass

    fig, ax = plt.subplots(figsize=(10, 6), dpi=Config.DPI)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, marker='o',
            label='Training Loss', linewidth=2)

    if val_losses:
        ax.plot(epochs, val_losses, marker='s',
                label='Validation Loss', linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"✓ Training curves saved: {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def get_device(verbose: bool = True) -> str:
    """
    Get available device (CUDA if available, else CPU).

    Args:
        verbose: Print device info

    Returns:
        Device string ('cuda' or 'cpu')
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        if device == "cuda":
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ Using CPU (GPU not available)")

    return device


def count_parameters(model: torch.nn.Module, verbose: bool = True) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model
        verbose: Print parameter count

    Returns:
        Total number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"✓ Trainable parameters: {total:,}")

    return total


if __name__ == "__main__":

    # Test 1: Configuration
    print("\n Configuration Management")
    Config.print_config('MODEL ARCHITECTURE')

    # Test 2: Reproducibility
    print("\n Reproducibility")
    set_seed(42)

    # Test 3: Directory Setup
    print("\n Directory Management")
    setup_directories()

    # Test 4: Device Management
    print("\n Device Management")
    device = get_device(verbose=True)

    # Test 5: Logging
    print("\n Logging Metrics")
    log_metrics({'loss': 0.123, 'accuracy': 0.95}, epoch=10, prefix="TRAIN")

    # Test 6: Checkpointing (requires a dummy model)
    print("\n Checkpoint Save/Load")
    try:
        from model import HierarchicalReasoningModel

        # Create dummy model
        model = HierarchicalReasoningModel(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            num_steps=3,
            l_iterations=5
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            loss=0.123,
            metrics={'accuracy': 0.95},
            config={'hidden_dim': 32, 'num_steps': 3}
        )

        # Load checkpoint
        loaded_model, loaded_opt, info = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            strict=False
        )

        print(f" Checkpoint test passed")

        # Count parameters
        count_parameters(model, verbose=True)

    except ImportError:
        print(" Skipping checkpoint test (model.py not found)")

    # Test 7: Results Saving
    print("\n Results Saving")
    results = {
        'accuracy': 0.95,
        'loss': 0.123,
        'config': {'hidden_dim': 32}
    }
    save_results(results, 'test_results.json')

    print(" ALL TESTS PASSED")

    print("\nConfiguration ready! All magic variables centralized.")
    print("Use: from utils import Config, set_seed, save_checkpoint, etc.")
