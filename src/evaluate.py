# src/evaluate.py
"""
Evaluation Script for Hierarchical Reasoning Model (HRM)
Evaluates the trained model checkpoint across CUDA, MPS, and CPU.
Steps:
1. Load model and checkpoint.
2. Evaluate on test dataset.
3. Compute MSE and MAE.
4. Optionally visualize predictions vs targets.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import HierarchicalReasoningModel
from dataset import get_dataloaders


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
    """Handles model loading and evaluation."""

    def __init__(
        self,
        checkpoint_path="hrm_checkpoint.pt",
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        num_steps=3,
    ):
        self.device = get_device()

        # Initialize model
        self.model = HierarchicalReasoningModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_steps=num_steps,
        ).to(self.device)

        # Load weights
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("✓ Model loaded successfully")

        # Load test data
        _, self.test_loader = get_dataloaders()

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        print(f"\nModel Configuration:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Reasoning steps: {num_steps}")
        print(f"  Test samples: {len(self.test_loader.dataset)}\n")

    def evaluate(self, verbose=False):
        """
        Evaluate the model on test data.

        Args:
            verbose: If True, print sample predictions

        Returns:
            tuple: (avg_mse, avg_mae)
        """
        total_mse, total_mae, total_samples = 0.0, 0.0, 0
        all_preds, all_targets = [], []

        print("Evaluating model...")
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                x = x.unsqueeze(-1)  # shape: [B, seq_len, 1]

                preds = self.model(x)

                mse_loss = self.mse(preds, y)
                mae_loss = self.mae(preds, y)

                total_mse += mse_loss.item() * x.size(0)
                total_mae += mae_loss.item() * x.size(0)
                total_samples += x.size(0)

                # Store for optional visualization
                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

                # Print sample predictions if verbose
                if verbose and batch_idx == 0:
                    print("\nSample Predictions (first batch):")
                    for i in range(min(5, len(preds))):
                        print(
                            f"  Target: {y[i].item():.4f} | Prediction: {preds[i].item():.4f}")

        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples

        print(f"\n{'='*50}")
        print(f"Evaluation Results:")
        print(f"{'='*50}")
        print(f"Mean Squared Error (MSE): {avg_mse:.6f}")
        print(f"Mean Absolute Error (MAE): {avg_mae:.6f}")
        print(f"Root Mean Squared Error (RMSE): {avg_mse**0.5:.6f}")
        print(f"{'='*50}\n")

        return avg_mse, avg_mae

    def evaluate_sample(self, sample_input):
        """
        Evaluate a single custom input.

        Args:
            sample_input: Tensor of shape [seq_len, input_dim] or [batch, seq_len, input_dim]

        Returns:
            prediction: Model output
        """
        self.model.eval()
        with torch.no_grad():
            if sample_input.dim() == 2:
                sample_input = sample_input.unsqueeze(0)  # Add batch dimension

            sample_input = sample_input.to(self.device)
            prediction = self.model(sample_input)

        return prediction


if __name__ == "__main__":
    # Create evaluator with correct hidden_dim (must match training config!)
    evaluator = Evaluator(
        checkpoint_path="hrm_checkpoint.pt",
        input_dim=1,
        hidden_dim=32,  # IMPORTANT: Must match the training configuration!
        output_dim=1,
        num_steps=3,
    )

    # Run evaluation with verbose output
    mse, mae = evaluator.evaluate(verbose=True)

    # Optional: Test on custom input
    print("\nTesting custom input:")
    # [seq_len=3, input_dim=1]
    custom_input = torch.tensor([[1.0], [2.0], [3.0]])
    prediction = evaluator.evaluate_sample(custom_input)
    print(f"Custom input: {custom_input.squeeze().tolist()}")
    print(f"Prediction: {prediction.item():.4f}")
