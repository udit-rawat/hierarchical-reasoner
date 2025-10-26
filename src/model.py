# src/model.py
"""
Hierarchical Reasoning Model (HRM) - Paper-Aligned Implementation
Based on arXiv:2506.21734

Key Features (NEW):
1. Hierarchical Convergence: L-module runs multiple iterations per H-step
2. L-module Reset: Resets between H-module updates
3. Multi-timescale Processing: Slow H-module, fast L-module
4. Adaptive Computation Time (ACT): Optional Q-learning for halting
5. Backward Compatible: Works with existing train.py and dataset.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighLevelReasoner(nn.Module):
    """
    High-Level Module (H-module): Slow, abstract planner
    Paper: Operates at θ rhythm (~4-8 Hz brain analogy)
    Updates infrequently, provides strategic control signals
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project input to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Slow planner (GRU for strategic reasoning)
        self.planner = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Control signal generator
        self.control_head = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, h_state=None):
        """
        Generate control signal for L-module

        Args:
            x: [batch, seq_len, input_dim] or [batch, hidden_dim]
            h_state: Previous hidden state [1, batch, hidden_dim]
        Returns:
            control_signal: [batch, hidden_dim]
            new_h_state: [1, batch, hidden_dim]
        """
        batch_size = x.size(0)

        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Project to hidden dim if needed
        if x.size(-1) != self.hidden_dim:
            x = self.input_projection(x)

        # Initialize hidden state if None
        if h_state is None:
            h_state = torch.zeros(
                1, batch_size, self.hidden_dim, device=x.device)

        # Run planner
        out, h_new = self.planner(x, h_state)

        # Generate control signal
        control = self.control_head(out[:, -1, :])
        control = self.norm(control)

        return control, h_new


class LowLevelExecutor(nn.Module):
    """
    Low-Level Module (L-module): Fast, detailed executor
    Paper: Operates at γ rhythm (~40 Hz brain analogy)
    Runs multiple iterations to reach local convergence
    KEY INNOVATION: Multi-iteration processing per H-step
    """

    def __init__(self, hidden_dim: int, num_iterations: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations

        # Fast recurrent processor
        self.executor = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Refinement network
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, control_signal):
        """
        Run multiple fast iterations to reach local convergence
        PAPER KEY CONCEPT: L-module iterates until convergence, then RESETS

        Args:
            control_signal: [batch, hidden_dim] - guidance from H-module
        Returns:
            converged_state: [batch, hidden_dim]
            trajectory: [batch, num_iterations, hidden_dim] - reasoning trace
        """
        batch_size = control_signal.size(0)

        # RESET: Initialize with control signal (paper's reset mechanism)
        state = control_signal.unsqueeze(1)  # [batch, 1, hidden_dim]
        h = None
        trajectory = []

        # Run fast iterations until local convergence
        for iteration in range(self.num_iterations):
            # Recurrent processing step
            out, h = self.executor(state, h)

            # Refinement and normalization
            refined = self.refiner(out[:, -1, :])
            refined = self.norm(refined)

            # Store trajectory for analysis
            trajectory.append(refined)

            # Update state for next iteration
            state = refined.unsqueeze(1)

        # Stack trajectory: [batch, num_iterations, hidden_dim]
        trajectory = torch.stack(trajectory, dim=1)

        # Final converged state
        converged_state = trajectory[:, -1, :]

        return converged_state, trajectory


class ACTModule(nn.Module):
    """
    Adaptive Computation Time (ACT) Module
    Paper: Q-learning based decision for halt/continue
    Optional: Can be disabled for simpler training
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [Q(halt), Q(continue)]
        )

    def forward(self, state):
        """
        Decide whether to halt or continue reasoning

        Args:
            state: [batch, hidden_dim]
        Returns:
            action: [batch] - 0 (halt) or 1 (continue)
            q_values: [batch, 2]
        """
        q_values = self.q_network(state)
        action = torch.argmax(q_values, dim=-1)  # Greedy policy
        return action, q_values


class HierarchicalReasoningModel(nn.Module):
    """
    Paper-Aligned Hierarchical Reasoning Model

    Changes from original implementation:
    1. L-module runs MULTIPLE iterations per H-step (not 1:1)
    2. L-module RESETS between H-steps
    3. Optional ACT for adaptive halting
    4. Returns reasoning trajectory for analysis

    Backward Compatible: Works with existing train.py
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_steps: int = 3,          # Number of H-module steps
        l_iterations: int = 5,        # NEW: L-module iterations per H-step
        use_act: bool = False,        # NEW: Enable adaptive computation
        min_h_steps: int = 1          # NEW: Min steps before ACT
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.l_iterations = l_iterations
        self.use_act = use_act
        self.min_h_steps = min_h_steps

        # Core modules
        self.high_level = HighLevelReasoner(input_dim, hidden_dim)
        self.low_level = LowLevelExecutor(hidden_dim, l_iterations)

        # Optional ACT module
        self.act_module = ACTModule(hidden_dim) if use_act else None

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, return_trajectory=False):
        """
        Hierarchical reasoning with paper-aligned multi-timescale processing

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_trajectory: If True, return reasoning trace
        Returns:
            final_output: [batch, output_dim]
            trajectory: Optional dict with reasoning steps
        """
        batch_size = x.size(0)

        # Initialize H-module state
        h_state = None
        current_input = x

        all_states = []
        reasoning_trajectory = []

        # Hierarchical reasoning loop
        for h_step in range(self.num_steps):
            # H-MODULE: Slow planning step (generates control signal)
            control_signal, h_state = self.high_level(current_input, h_state)

            # L-MODULE: Fast execution with MULTIPLE iterations
            # KEY: L-module runs l_iterations times, then RESETS
            l_state, l_trajectory = self.low_level(control_signal)

            # Store states
            all_states.append(l_state)

            if return_trajectory:
                reasoning_trajectory.append({
                    'h_step': h_step,
                    'control_signal': control_signal.detach(),
                    'l_trajectory': l_trajectory.detach(),
                    'final_l_state': l_state.detach()
                })

            # ACT: Decide whether to halt or continue
            if self.use_act and h_step >= self.min_h_steps:
                action, q_values = self.act_module(l_state)

                # If all samples want to halt, stop
                if action.sum() == 0:
                    break

            # Prepare input for next H-module step
            # Use converged L-state as input
            current_input = l_state.unsqueeze(1)

        # Aggregate reasoning steps (use final state)
        final_state = all_states[-1]

        # Generate output
        output = self.output_head(final_state)

        if return_trajectory:
            trajectory_info = {
                'num_h_steps': len(all_states),
                'l_iterations_per_step': self.l_iterations,
                'total_l_iterations': len(all_states) * self.l_iterations,
                'steps': reasoning_trajectory
            }
            return output, trajectory_info

        return output


if __name__ == "__main__":
    print("="*70)
    print(" HRM: Testing Multi-Timescale Hierarchical Reasoning")
    print("="*70)

    # Test configuration matching dataset.py
    batch_size = 4
    seq_len = 3
    input_dim = 1
    hidden_dim = 32
    output_dim = 1

    # Create model
    model = HierarchicalReasoningModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_steps=3,           # 3 H-module steps
        l_iterations=5,        # 5 L-module iterations per H-step
        use_act=False          # Disable ACT for now
    )

    # Test input
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass with trajectory
    output, trajectory = model(dummy_input, return_trajectory=True)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nHierarchical Reasoning Breakdown:")
    print(f"  Total H-module steps: {trajectory['num_h_steps']}")
    print(f"  L-iterations per H-step: {trajectory['l_iterations_per_step']}")
    print(f"  Total L-iterations: {trajectory['total_l_iterations']}")

    for step in trajectory['steps']:
        l_traj = step['l_trajectory']
        print(f"  H-step {step['h_step']}:")
        print(f"    Control signal shape: {step['control_signal'].shape}")
        print(f"    L-trajectory shape: {l_traj.shape}")
        print(f"    → L-module ran {l_traj.shape[1]} fast iterations")
