# src/model.py
"""
Hierarchical Reasoning Model (HRM)
Minimal PyTorch implementation of a hierarchical reasoning system.
Structure:
- HighLevelReasoner: Plans reasoning steps (like a controller)
- LowLevelExecutor: Executes sub-tasks (like a solver)
- HierarchicalReasoningModel: Combines both for end-to-end reasoning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighLevelReasoner(nn.Module):
    """Plans reasoning steps or selects which submodule to use next."""

    def __init__(self, input_dim: int, hidden_dim: int, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        # Project input to hidden_dim first
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # Controller processes hidden_dim throughout
        self.controller = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, input_dim]
        Returns:
            step_controls: Tensor of shape [batch, num_steps, hidden_dim]
        """
        batch_size = x.size(0)
        outputs = []
        h = torch.zeros(
            1, batch_size, self.controller.hidden_size, device=x.device
        )

        # Project input to hidden dimension
        # [batch, seq_len, input_dim] -> [batch, seq_len, hidden_dim]
        inp = self.input_projection(x)

        for _ in range(self.num_steps):
            out, h = self.controller(inp, h)
            control_signal = self.policy_head(out[:, -1, :])
            outputs.append(control_signal)
            inp = control_signal.unsqueeze(1)  # [batch, 1, hidden_dim]

        return torch.stack(outputs, dim=1)


class LowLevelExecutor(nn.Module):
    """Executes individual reasoning substeps given control signals."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.executor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, control_signals):
        """
        Args:
            control_signals: [batch, num_steps, hidden_dim]
        Returns:
            step_outputs: [batch, num_steps, output_dim]
        """
        batch, steps, dim = control_signals.shape
        flat = control_signals.reshape(batch * steps, dim)
        out = self.executor(flat)
        return out.reshape(batch, steps, -1)


class HierarchicalReasoningModel(nn.Module):
    """Combines high-level planner and low-level executor."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_steps: int = 3):
        super().__init__()
        self.high_level = HighLevelReasoner(input_dim, hidden_dim, num_steps)
        self.low_level = LowLevelExecutor(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]
        Returns:
            final_output: [batch, output_dim]
        """
        control_signals = self.high_level(x)
        step_outputs = self.low_level(control_signals)
        # Aggregate over reasoning steps (simple mean pooling)
        final_output = step_outputs.mean(dim=1)
        return final_output


if __name__ == "__main__":
    # Sanity check
    print("Testing HierarchicalReasoningModel...")

    # Test 1: Original dimensions
    model1 = HierarchicalReasoningModel(
        input_dim=64, hidden_dim=64, output_dim=10, num_steps=3
    )
    dummy_input1 = torch.randn(8, 5, 64)
    out1 = model1(dummy_input1)
    print(f"Test 1 - Input: {dummy_input1.shape}, Output: {out1.shape}")

    # Test 2: Small dimensions (like train.py)
    model2 = HierarchicalReasoningModel(
        input_dim=1, hidden_dim=32, output_dim=1, num_steps=3
    )
    dummy_input2 = torch.randn(4, 3, 1)  # batch=4, seq_len=3, input_dim=1
    out2 = model2(dummy_input2)
    print(f"Test 2 - Input: {dummy_input2.shape}, Output: {out2.shape}")

    # Test 3: Different dimensions
    model3 = HierarchicalReasoningModel(
        input_dim=10, hidden_dim=128, output_dim=5, num_steps=5
    )
    dummy_input3 = torch.randn(16, 7, 10)
    out3 = model3(dummy_input3)
    print(f"Test 3 - Input: {dummy_input3.shape}, Output: {out3.shape}")

    print("\nAll tests passed! ✓")
