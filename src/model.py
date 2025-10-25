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
        self.controller = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch, seq_len, input_dim]
        Returns:
            step_controls: List of hidden states for each reasoning step
        """
        batch_size = x.size(0)
        outputs = []

        h = torch.zeros(
            1, batch_size, self.controller.hidden_size, device=x.device)
        inp = x

        for _ in range(self.num_steps):
            out, h = self.controller(inp, h)
            control_signal = self.policy_head(out[:, -1, :])
            outputs.append(control_signal)
            inp = control_signal.unsqueeze(1)
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
    model = HierarchicalReasoningModel(
        input_dim=64, hidden_dim=64, output_dim=10, num_steps=3)
    dummy_input = torch.randn(8, 5, 64)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # [8, 10]
