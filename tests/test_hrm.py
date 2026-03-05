"""
pytest suite for Hierarchical Reasoning Model
Run: pytest tests/ -v
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from torch.utils.data import DataLoader
from src.model import HierarchicalReasoningModel, HighLevelReasoner, LowLevelExecutor
from src.dataset import SyntheticReasoningDataset, get_dataloaders
from src.datasetSudoku import SudokuDataset
from src.datasetMaze import MazeDataset


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def default_model():
    return HierarchicalReasoningModel(
        input_dim=1, hidden_dim=32, output_dim=1,
        num_steps=3, l_iterations=5, use_act=False
    )

@pytest.fixture
def dummy_input():
    return torch.randn(4, 3, 1)  # batch=4, seq=3, input_dim=1


# ─── Model Tests ─────────────────────────────────────────────────────────────

class TestModel:

    def test_output_shape(self, default_model, dummy_input):
        out = default_model(dummy_input)
        assert out.shape == (4, 1)

    def test_output_is_finite(self, default_model, dummy_input):
        out = default_model(dummy_input)
        assert torch.isfinite(out).all()

    def test_trajectory_keys(self, default_model, dummy_input):
        _, traj = default_model(dummy_input, return_trajectory=True)
        assert 'num_h_steps' in traj
        assert 'l_iterations_per_step' in traj
        assert 'total_l_iterations' in traj
        assert 'steps' in traj

    def test_trajectory_depth(self, default_model, dummy_input):
        _, traj = default_model(dummy_input, return_trajectory=True)
        assert traj['num_h_steps'] == 3
        assert traj['l_iterations_per_step'] == 5
        assert traj['total_l_iterations'] == 15

    def test_trajectory_step_shapes(self, default_model, dummy_input):
        _, traj = default_model(dummy_input, return_trajectory=True)
        for step in traj['steps']:
            assert step['l_trajectory'].shape == (4, 5, 32)
            assert step['control_signal'].shape == (4, 32)

    @pytest.mark.parametrize("num_steps,l_iters", [
        (1, 1), (1, 5), (1, 10),
        (3, 1), (3, 5), (3, 10),
        (5, 1), (5, 5), (5, 10),
    ])
    def test_hyperparameter_configs(self, num_steps, l_iters, dummy_input):
        model = HierarchicalReasoningModel(
            input_dim=1, hidden_dim=32, output_dim=1,
            num_steps=num_steps, l_iterations=l_iters
        )
        out = model(dummy_input)
        assert out.shape == (4, 1)
        assert torch.isfinite(out).all()

    def test_param_count_increases_with_hidden_dim(self):
        m32 = HierarchicalReasoningModel(1, 32, 1)
        m64 = HierarchicalReasoningModel(1, 64, 1)
        assert sum(p.numel() for p in m64.parameters()) > \
               sum(p.numel() for p in m32.parameters())

    def test_h_module_output_shape(self):
        h = HighLevelReasoner(input_dim=1, hidden_dim=32)
        x = torch.randn(4, 1, 1)
        ctrl, state = h(x)
        assert ctrl.shape == (4, 32)
        assert state.shape == (1, 4, 32)

    def test_l_module_output_shape(self):
        l = LowLevelExecutor(hidden_dim=32, num_iterations=5)
        ctrl = torch.randn(4, 32)
        out, traj = l(ctrl)
        assert out.shape == (4, 32)
        assert traj.shape == (4, 5, 32)

    def test_no_gradient_explosion(self, default_model, dummy_input):
        out = default_model(dummy_input)
        loss = out.sum()
        loss.backward()
        for p in default_model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


# ─── Dataset Tests ────────────────────────────────────────────────────────────

class TestArithmeticDataset:

    def test_length(self):
        ds = SyntheticReasoningDataset(size=100)
        assert len(ds) == 100

    def test_sample_shapes(self):
        ds = SyntheticReasoningDataset(size=10)
        x, y = ds[0]
        assert x.shape == (3,)
        assert y.shape == (1,)

    def test_dataloader_batch_shape(self):
        train, _ = get_dataloaders(batch_size=8, train_size=100, test_size=20)
        x, y = next(iter(train))
        assert x.shape == (8, 3)
        assert y.shape == (8, 1)

    def test_op_token_is_binary(self):
        ds = SyntheticReasoningDataset(size=200)
        for i in range(len(ds)):
            x, _ = ds[i]
            assert x[2].item() in (0.0, 1.0)


class TestSudokuDataset:

    @pytest.fixture(scope='class')
    def sudoku_4x4(self):
        return SudokuDataset(num_samples=10, grid_size=2, difficulty=0.3, seed=42)

    def test_length(self, sudoku_4x4):
        assert len(sudoku_4x4) == 10

    def test_sample_shapes(self, sudoku_4x4):
        puzzle, solution = sudoku_4x4[0]
        assert puzzle.shape == (16,)
        assert solution.shape == (16,)

    def test_value_range(self, sudoku_4x4):
        puzzle, solution = sudoku_4x4[0]
        assert puzzle.min() >= 0.0
        assert puzzle.max() <= 1.0
        assert solution.min() >= 0.0
        assert solution.max() <= 1.0

    def test_dataloader_compatible(self, sudoku_4x4):
        loader = DataLoader(sudoku_4x4, batch_size=4)
        x, y = next(iter(loader))
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)


class TestMazeDataset:

    @pytest.fixture(scope='class')
    def maze_5x5(self):
        return MazeDataset(num_samples=20, maze_size=5, seed=42)

    def test_length(self, maze_5x5):
        assert len(maze_5x5) == 20

    def test_sample_shapes(self, maze_5x5):
        maze, solution = maze_5x5[0]
        assert maze.shape == (25,)
        assert solution.shape == (25,)

    def test_start_encoded(self, maze_5x5):
        # Start cell (index 0) should be 0.5
        maze, _ = maze_5x5[0]
        assert maze[0].item() == pytest.approx(0.5)

    def test_goal_encoded(self, maze_5x5):
        # Goal cell (index 24) should be 0.75
        maze, _ = maze_5x5[0]
        assert maze[24].item() == pytest.approx(0.75)

    def test_solution_has_path(self, maze_5x5):
        # Solution must have at least one path cell (0.25)
        _, solution = maze_5x5[0]
        path_cells = (solution == 0.25).sum().item()
        assert path_cells > 0

    def test_dataloader_compatible(self, maze_5x5):
        loader = DataLoader(maze_5x5, batch_size=4)
        x, y = next(iter(loader))
        assert x.shape == (4, 25)
        assert y.shape == (4, 25)

    def test_7x7_maze(self):
        ds = MazeDataset(num_samples=5, maze_size=7, seed=42)
        maze, sol = ds[0]
        assert maze.shape == (49,)
        assert sol.shape == (49,)


# ─── Checkpoint Tests ─────────────────────────────────────────────────────────

class TestCheckpoint:

    def test_save_and_load(self, default_model, dummy_input, tmp_path):
        path = tmp_path / "test_ckpt.pt"
        torch.save({'model_state_dict': default_model.state_dict()}, path)

        loaded = HierarchicalReasoningModel(
            input_dim=1, hidden_dim=32, output_dim=1,
            num_steps=3, l_iterations=5
        )
        ckpt = torch.load(path, map_location='cpu')
        loaded.load_state_dict(ckpt['model_state_dict'])

        out_orig = default_model(dummy_input)
        out_loaded = loaded(dummy_input)
        assert torch.allclose(out_orig, out_loaded, atol=1e-6)

    def test_weights_unchanged_after_load(self, default_model, tmp_path):
        path = tmp_path / "test_ckpt.pt"
        torch.save({'model_state_dict': default_model.state_dict()}, path)

        loaded = HierarchicalReasoningModel(1, 32, 1, num_steps=3, l_iterations=5)
        ckpt = torch.load(path, map_location='cpu')
        loaded.load_state_dict(ckpt['model_state_dict'])

        for (n1, p1), (n2, p2) in zip(
            default_model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"


# ─── Metrics Tests ────────────────────────────────────────────────────────────

class TestMetrics:

    def test_mse_non_negative(self, default_model, dummy_input):
        criterion = torch.nn.MSELoss()
        target = torch.randn(4, 1)
        out = default_model(dummy_input)
        loss = criterion(out, target)
        assert loss.item() >= 0.0

    def test_mae_non_negative(self, default_model, dummy_input):
        criterion = torch.nn.L1Loss()
        target = torch.randn(4, 1)
        out = default_model(dummy_input)
        loss = criterion(out, target)
        assert loss.item() >= 0.0

    def test_zero_loss_on_perfect_prediction(self):
        model = HierarchicalReasoningModel(1, 32, 1, num_steps=3, l_iterations=5)
        x = torch.randn(4, 3, 1)
        out = model(x)
        criterion = torch.nn.MSELoss()
        loss = criterion(out, out.detach())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
