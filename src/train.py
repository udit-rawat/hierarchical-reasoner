# src/train.py
"""
Training Script for Hierarchical Reasoning Model (HRM)
------------------------------------------------------
Universal version supporting CUDA, Metal (MPS), and CPU.
Steps:
1. Load dataset from dataset.py
2. Initialize model from model.py
3. Define optimizer and loss
4. Train and evaluate
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import HierarchicalReasoningModel
from dataset import get_dataloaders


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
    """Encapsulates model training and evaluation."""

    def __init__(
        self,
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        num_steps=3,
        lr=1e-3,
        epochs=50,
    ):
        self.device = get_device()
        self.model = HierarchicalReasoningModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_steps=num_steps,
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.train_loader, self.test_loader = get_dataloaders()

        print(f"\nModel Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Reasoning steps: {num_steps}")
        print(
            f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        loop = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            leave=False
        )

        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device)

            # Reshape input to [batch, seq_len, input_dim]
            x = x.unsqueeze(-1)  # [B, 3] → [B, 3, 1]

            pred = self.model(x)
            loss = self.criterion(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                x = x.unsqueeze(-1)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(self.test_loader.dataset)
        return avg_loss

    def train(self):
        print("Starting training...")
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.evaluate()

            print(
                f"[Epoch {epoch+1:3d}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "hrm_checkpoint.pt")
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")

        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: hrm_checkpoint.pt")


if __name__ == "__main__":
    # Default configuration
    trainer = Trainer(
        input_dim=1,      # Input features per timestep
        hidden_dim=32,    # Hidden dimension for reasoning
        output_dim=1,     # Output prediction dimension
        num_steps=3,      # Number of hierarchical reasoning steps
        lr=1e-3,          # Learning rate
        epochs=50         # Training epochs
    )
    trainer.train()
