# src/evaluate.py
"""
Evaluation Script for Hierarchical Reasoning Model (HRM)
--------------------------------------------------------
Paper-Aligned Evaluation with Trajectory Analysis
Evaluates the trained model checkpoint across CUDA, MPS, and CPU.

NEW FEATURES:
1. Trajectory analysis: Visualize hierarchical reasoning steps
2. Computation depth metrics: Track H-module and L-module usage
3. Per-step error analysis: See how reasoning improves across steps
4. Enhanced checkpoint loading: Handles full training state
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import HierarchicalReasoningModel
from dataset import get_dataloaders
import numpy as np


def get_device():
    """Auto-select best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple Metal (MPS)"
    else:
        device = torch.device("cpu")
        name = "CPU"

    print(f"Using device: {name}")
    return device


class Evaluator:
    """
    Handles model loading and evaluation.

    NEW: Supports paper-aligned HRM with trajectory analysis
    """

    def __init__(
        self,
        checkpoint_path="hrm_checkpoint.pt",
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        num_steps=3,
        l_iterations=5,        # NEW: Must match training config
        use_act=False,         # NEW: Must match training config
    ):
        self.device = get_device()
        self.num_steps = num_steps
        self.l_iterations = l_iterations

        # Initialize paper-aligned model
        self.model = HierarchicalReasoningModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_steps=num_steps,
            l_iterations=l_iterations,  # NEW parameter
            use_act=use_act,            # NEW parameter
        ).to(self.device)

        # Load checkpoint (handle both old and new formats)
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(
                f"  Training loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
            print(
                f"  Validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        else:
            # Backward compatibility: old format (just state dict)
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("✓ Model loaded successfully")

        # Load test data
        _, self.test_loader = get_dataloaders()

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  H-module steps: {num_steps}")
        print(f"  L-module iterations per H-step: {l_iterations}")  # NEW
        print(f"  Total computation depth: {num_steps * l_iterations}")  # NEW
        print(f"  Adaptive computation (ACT): {use_act}")  # NEW
        print(f"  Test samples: {len(self.test_loader.dataset)}")

    def evaluate(self, verbose=False, analyze_trajectory=True):
        """
        Evaluate the model on test data.

        Args:
            verbose: If True, print sample predictions
            analyze_trajectory: If True, analyze reasoning trajectories (NEW)

        Returns:
            dict: Comprehensive evaluation results
        """
        total_mse, total_mae, total_samples = 0.0, 0.0, 0
        all_preds, all_targets = [], []

        # NEW: Trajectory analysis storage
        trajectory_stats = {
            'h_steps_used': [],
            'total_l_iterations': [],
            'computation_depths': []
        }

        print("Evaluating model...")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                x = x.unsqueeze(-1)  # shape: [B, seq_len, 1]

                # NEW: Get predictions with trajectory for first batch
                if analyze_trajectory and batch_idx == 0:
                    preds, trajectory = self.model(x, return_trajectory=True)

                    # Store trajectory statistics
                    trajectory_stats['h_steps_used'].append(
                        trajectory['num_h_steps'])
                    trajectory_stats['total_l_iterations'].append(
                        trajectory['total_l_iterations'])
                    trajectory_stats['computation_depths'].append(
                        trajectory['num_h_steps'] *
                        trajectory['l_iterations_per_step']
                    )

                    # Print detailed trajectory for first sample
                    if verbose:
                        self._print_trajectory_details(
                            trajectory, x[0], y[0], preds[0])
                else:
                    preds = self.model(x)

                mse_loss = self.mse(preds, y)
                mae_loss = self.mae(preds, y)

                total_mse += mse_loss.item() * x.size(0)
                total_mae += mae_loss.item() * x.size(0)
                total_samples += x.size(0)

                # Store predictions and targets
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

                # Print sample predictions if verbose
                if verbose and batch_idx == 0:
                    print("\nSample Predictions (first batch):")
                    for i in range(min(5, len(preds))):
                        error = abs(y[i].item() - preds[i].item())
                        print(
                            f"  Sample {i+1}: Target: {y[i].item():.4f} | "
                            f"Prediction: {preds[i].item():.4f} | "
                            f"Error: {error:.4f}"
                        )

        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples

        # Compute additional metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # NEW: Compute correlation and R²
        correlation = np.corrcoef(
            all_targets.numpy().flatten(),
            all_preds.numpy().flatten()
        )[0, 1]

        ss_res = ((all_targets - all_preds) ** 2).sum().item()
        ss_tot = ((all_targets - all_targets.mean()) ** 2).sum().item()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Print results

        print(f"Mean Squared Error (MSE):      {avg_mse:.6f}")
        print(f"Mean Absolute Error (MAE):     {avg_mae:.6f}")
        print(f"Root Mean Squared Error (RMSE): {avg_mse**0.5:.6f}")
        print(f"Correlation (Pearson):         {correlation:.6f}")  # NEW
        print(f"R² Score:                      {r_squared:.6f}")     # NEW

        # NEW: Trajectory analysis summary
        if trajectory_stats['h_steps_used']:
            print(f"\nHierarchical Reasoning Analysis:")
            print(
                f"  Average H-steps used: {np.mean(trajectory_stats['h_steps_used']):.2f}")
            print(
                f"  Average L-iterations: {np.mean(trajectory_stats['total_l_iterations']):.2f}")
            print(
                f"  Average computation depth: {np.mean(trajectory_stats['computation_depths']):.2f}")
            print(
                f"  Max computation depth: {max(trajectory_stats['computation_depths'])}")

        return {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': avg_mse**0.5,
            'correlation': correlation,
            'r_squared': r_squared,
            'predictions': all_preds,
            'targets': all_targets,
            'trajectory_stats': trajectory_stats
        }

    def _print_trajectory_details(self, trajectory, input_sample, target, prediction):
        """
        NEW: Print detailed trajectory information for a sample
        """

        print(f"Input: {input_sample.squeeze().cpu().tolist()}")
        print(f"Target: {target.item():.4f}")
        print(f"Prediction: {prediction.item():.4f}")
        print(f"Error: {abs(target.item() - prediction.item()):.4f}")
        print(f"\nReasoning Steps:")
        print(f"  Total H-module steps: {trajectory['num_h_steps']}")
        print(
            f"  L-iterations per H-step: {trajectory['l_iterations_per_step']}")
        print(f"  Total computation depth: {trajectory['total_l_iterations']}")

        print(f"\nStep-by-step Breakdown:")
        for i, step in enumerate(trajectory['steps']):
            print(f"  H-step {i+1}:")
            print(f"    Control signal shape: {step['control_signal'].shape}")
            print(f"    L-module trajectory: {step['l_trajectory'].shape}")
            print(
                f"    L-iterations executed: {step['l_trajectory'].shape[1]}")

            # Show L-module convergence (first few dimensions)
            l_traj = step['l_trajectory'][0, :,
                                          :3].cpu().numpy()  # First 3 dims
            print(f"    L-module convergence (first 3 dims):")
            for l_iter in range(min(3, l_traj.shape[0])):
                print(f"      Iteration {l_iter+1}: {l_traj[l_iter]}")

    def evaluate_sample(self, sample_input, show_trajectory=True):
        """
        Evaluate a single custom input.

        Args:
            sample_input: Tensor of shape [seq_len, input_dim] or [batch, seq_len, input_dim]
            show_trajectory: If True, return reasoning trajectory (NEW)

        Returns:
            prediction: Model output
            trajectory: Optional reasoning trace (if show_trajectory=True)
        """
        self.model.eval()
        with torch.no_grad():
            if sample_input.dim() == 2:
                sample_input = sample_input.unsqueeze(0)  # Add batch dimension

            sample_input = sample_input.to(self.device)

            if show_trajectory:
                prediction, trajectory = self.model(
                    sample_input, return_trajectory=True)
                return prediction, trajectory
            else:
                prediction = self.model(sample_input)
                return prediction

    def compare_reasoning_depths(self, test_samples=10):
        """
        NEW: Compare model performance with different reasoning depths
        Useful for understanding the value of hierarchical processing
        """

        # Get some test samples
        test_batch = next(iter(self.test_loader))
        x, y = test_batch[0][:test_samples].to(
            self.device), test_batch[1][:test_samples].to(self.device)
        x = x.unsqueeze(-1)

        with torch.no_grad():
            preds, trajectory = self.model(x, return_trajectory=True)

            print(f"Test samples: {test_samples}")
            print(f"H-steps used: {trajectory['num_h_steps']}")
            print(
                f"L-iterations per H-step: {trajectory['l_iterations_per_step']}")
            print(
                f"Total computation depth: {trajectory['total_l_iterations']}")

            # Compute error
            mse = self.mse(preds, y).item()
            mae = self.mae(preds, y).item()

            print(f"\nPerformance with full depth:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")


if __name__ == "__main__":
    # Create evaluator with paper-aligned configuration
    # IMPORTANT: Parameters must match training configuration!
    evaluator = Evaluator(
        checkpoint_path="hrm_checkpoint.pt",
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        num_steps=3,
        l_iterations=5,    # NEW: Must match training
        use_act=True,     # NEW: Must match training
    )

    # Run comprehensive evaluation
    results = evaluator.evaluate(verbose=True, analyze_trajectory=True)

    # NEW: Compare reasoning depths

    evaluator.compare_reasoning_depths(test_samples=10)

    # Test on custom input with trajectory
    print("\nTesting custom input with trajectory:")
    # [seq_len=3, input_dim=1]
    custom_input = torch.tensor([[1.0], [2.0], [3.0]])
    prediction, trajectory = evaluator.evaluate_sample(
        custom_input, show_trajectory=True)

    print(f"Custom input: {custom_input.squeeze().tolist()}")
    print(f"Prediction: {prediction.item():.4f}")
    print(f"Reasoning steps used: {trajectory['num_h_steps']}")
    print(f"Total L-iterations: {trajectory['total_l_iterations']}")
