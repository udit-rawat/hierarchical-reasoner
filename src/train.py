# src/train.py
"""
Training Script for Hierarchical Reasoning Model (HRM)
------------------------------------------------------
Paper-Aligned Training with Multi-Timescale Processing
Universal version supporting CUDA, Metal (MPS), and CPU.

NEW FEATURES:
1. Supports L-module multi-iteration training
2. Optional trajectory logging for analysis
3. Adaptive learning rate scheduling
4. Enhanced metrics tracking
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import HierarchicalReasoningModel
from dataset import get_dataloaders
import time


def get_device():
    """Auto-selects the best available device (CUDA, MPS, or CPU)."""
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


class Trainer:
    """
    Encapsulates model training and evaluation.

    NEW: Supports paper-aligned HRM with multi-iteration L-module
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        num_steps=3,
        l_iterations=5,        # NEW: L-module iterations per H-step
        use_act=False,         # NEW: Adaptive computation time
        lr=1e-3,
        epochs=50,
        use_scheduler=False,   # NEW: Learning rate scheduling
        log_trajectory=False,  # NEW: Log reasoning trajectories
    ):
        self.device = get_device()
        self.log_trajectory = log_trajectory

        # Initialize paper-aligned model
        self.model = HierarchicalReasoningModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_steps=num_steps,
            l_iterations=l_iterations,  # NEW parameter
            use_act=use_act,            # NEW parameter
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

        # NEW: Optional learning rate scheduler
        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

        self.train_loader, self.test_loader = get_dataloaders()

        # Print enhanced model info
        print(f"\n{'='*70}")
        print(f"Paper-Aligned HRM Configuration:")
        print(f"{'='*70}")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  H-module steps: {num_steps}")
        print(f"  L-module iterations per H-step: {l_iterations}")  # NEW
        print(f"  Total computation depth: {num_steps * l_iterations}")  # NEW
        print(f"  Adaptive computation (ACT): {use_act}")  # NEW
        print(
            f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Learning rate: {lr}")
        print(f"  LR scheduling: {use_scheduler}")
        print(f"{'='*70}\n")

    def train_one_epoch(self, epoch):
        """
        Train for one epoch.

        NEW: Optionally logs reasoning trajectories for analysis
        """
        self.model.train()
        total_loss = 0.0
        loop = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            leave=False
        )

        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(self.device), y.to(self.device)

            # Reshape input to [batch, seq_len, input_dim]
            x = x.unsqueeze(-1)  # [B, 3] → [B, 3, 1]

            # Forward pass
            if self.log_trajectory and batch_idx == 0:
                # Log trajectory for first batch (for analysis)
                pred, trajectory = self.model(x, return_trajectory=True)

                # Optional: Print trajectory info for debugging
                if epoch % 10 == 0:  # Every 10 epochs
                    print(f"\n  Trajectory: {trajectory['num_h_steps']} H-steps, "
                          f"{trajectory['total_l_iterations']} L-iterations")
            else:
                pred = self.model(x)

            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()

            # Optional: Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss

    def evaluate(self, return_detailed=False):
        """
        Evaluate model on test set.

        NEW: Can return detailed metrics including trajectory analysis
        """
        self.model.eval()
        total_loss = 0.0

        # NEW: Track additional metrics
        predictions = []
        targets = []
        trajectories = []

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                x = x.unsqueeze(-1)

                # Forward pass with optional trajectory
                if return_detailed and batch_idx == 0:
                    pred, trajectory = self.model(x, return_trajectory=True)
                    trajectories.append(trajectory)
                else:
                    pred = self.model(x)

                loss = self.criterion(pred, y)
                total_loss += loss.item() * x.size(0)

                if return_detailed:
                    predictions.append(pred.cpu())
                    targets.append(y.cpu())

        avg_loss = total_loss / len(self.test_loader.dataset)

        if return_detailed:
            return {
                'loss': avg_loss,
                'predictions': torch.cat(predictions),
                'targets': torch.cat(targets),
                'trajectories': trajectories
            }

        return avg_loss

    def train(self):
        """
        Full training loop with enhanced logging.

        NEW: Tracks training time, best metrics, and optional trajectory analysis
        """
        print("Starting paper-aligned HRM training...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}\n")

        best_val_loss = float('inf')
        best_epoch = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Evaluate
            val_loss = self.evaluate()

            epoch_time = time.time() - epoch_start

            # Print progress
            print(
                f"[Epoch {epoch+1:3d}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, "hrm_checkpoint.pt")
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")

            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

        total_time = time.time() - start_time

        # Final evaluation with detailed metrics
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Model saved to: hrm_checkpoint.pt")

        # NEW: Detailed final evaluation
        print(f"\nRunning detailed final evaluation...")
        detailed_results = self.evaluate(return_detailed=True)

        print(f"Final Test Loss: {detailed_results['loss']:.6f}")

        if detailed_results['trajectories']:
            traj = detailed_results['trajectories'][0]
            print(f"\nReasoning Analysis (sample batch):")
            print(f"  H-module steps taken: {traj['num_h_steps']}")
            print(
                f"  L-iterations per H-step: {traj['l_iterations_per_step']}")
            print(f"  Total computation depth: {traj['total_l_iterations']}")

        print(f"{'='*70}\n")


if __name__ == "__main__":

    trainer = Trainer(
        input_dim=1,          # Input features per timestep
        hidden_dim=32,        # Hidden dimension for reasoning
        output_dim=1,         # Output prediction dimension
        num_steps=3,          # Number of H-module steps (slow planning)
        # NEW: L-module iterations per H-step (fast execution)
        l_iterations=5,
        use_act=True,        # NEW: Adaptive computation (set True to enable)
        lr=1e-3,              # Learning rate
        epochs=50,            # Training epochs
        use_scheduler=False,  # NEW: Learning rate scheduling
        log_trajectory=True,  # NEW: Log reasoning trajectories
    )
    trainer.train()
