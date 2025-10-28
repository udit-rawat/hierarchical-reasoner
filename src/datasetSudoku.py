"""
Sudoku Dataset for Hierarchical Reasoning Model

Generates Sudoku puzzles using py-sudoku library and formats them for HRM training.
Paper uses 1000 training samples for near-perfect Sudoku solving.

Installation:
    pip install py-sudoku

Usage:
    train_dataset = SudokuDataset(num_samples=1000, grid_size=3, difficulty=0.5)
    test_dataset = SudokuDataset(num_samples=200, grid_size=3, difficulty=0.7)
"""

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    from sudoku import Sudoku
except ImportError:
    print("ERROR: py-sudoku not installed. Run: pip install py-sudoku")
    raise


class SudokuDataset(Dataset):
    """
    Sudoku puzzle dataset for training HRM.

    Args:
        num_samples (int): Number of puzzles to generate
        grid_size (int): Size of Sudoku grid (3 = 9×9, 2 = 4×4)
        difficulty (float): Difficulty level 0.0-1.0 (higher = harder)
        seed (int): Random seed for reproducibility
        normalize (bool): Normalize values to [0,1] range

    Returns:
        puzzle (Tensor): Flattened puzzle grid with 0 for empty cells [81] for 9×9
        solution (Tensor): Flattened complete solution grid [81] for 9×9
    """

    def __init__(self, num_samples=1000, grid_size=3, difficulty=0.5, seed=None, normalize=True):
        super().__init__()
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.full_size = (grid_size ** 2) ** 2  # 9×9 = 81 cells
        self.difficulty = difficulty
        self.normalize = normalize

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        print(
            f"Generating {num_samples} Sudoku puzzles ({grid_size**2}×{grid_size**2})...")
        self.puzzles = []
        self.solutions = []

        for i in range(num_samples):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} puzzles")

            # Generate puzzle
            puzzle_obj = Sudoku(grid_size, seed=np.random.randint(0, 1000000))
            puzzle_obj = puzzle_obj.difficulty(difficulty)

            # Get puzzle and solution grids
            puzzle_grid = puzzle_obj.board
            solution_grid = puzzle_obj.solve().board

            # Convert to tensors
            puzzle_tensor = self._grid_to_tensor(puzzle_grid)
            solution_tensor = self._grid_to_tensor(solution_grid)

            self.puzzles.append(puzzle_tensor)
            self.solutions.append(solution_tensor)

        print(f" Dataset ready: {num_samples} puzzles")
        self._print_stats()

    def _grid_to_tensor(self, grid):
        """
        Convert 2D Sudoku grid to 1D tensor.

        Args:
            grid: 2D list (9×9) with None for empty cells

        Returns:
            Tensor: Flattened grid [81] with 0 for empty cells
        """
        flat_grid = []
        for row in grid:
            for cell in row:
                # None means empty cell, replace with 0
                flat_grid.append(0 if cell is None else cell)

        tensor = torch.tensor(flat_grid, dtype=torch.float32)

        # Normalize to [0, 1] if requested
        if self.normalize:
            # Divide by max value (9 for 9×9)
            tensor = tensor / (self.grid_size ** 2)

        return tensor

    def _print_stats(self):
        """Print dataset statistics"""
        sample_puzzle = self.puzzles[0]
        sample_solution = self.solutions[0]

        # Count empty cells in first puzzle
        empty_cells = (sample_puzzle == 0).sum().item()
        total_cells = len(sample_puzzle)
        filled_ratio = 1 - (empty_cells / total_cells)

        print(f"\nDataset Statistics:")
        print(f"  Total samples: {self.num_samples}")
        print(f"  Grid size: {self.grid_size**2}×{self.grid_size**2}")
        print(f"  Input dimension: {self.full_size}")
        print(f"  Output dimension: {self.full_size}")
        print(f"  Difficulty: {self.difficulty:.2f}")
        print(
            f"  Sample filled ratio: {filled_ratio:.2%} ({total_cells - empty_cells}/{total_cells} cells)")
        print(
            f"  Value range: [0, {1 if self.normalize else self.grid_size**2}]")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            puzzle (Tensor): Input puzzle with empty cells as 0
            solution (Tensor): Complete solution
        """
        return self.puzzles[idx], self.solutions[idx]

    def visualize_sample(self, idx=0):
        """
        Print a sample puzzle and solution in readable format.

        Args:
            idx: Index of sample to visualize
        """
        puzzle = self.puzzles[idx]
        solution = self.solutions[idx]

        # Denormalize if needed
        if self.normalize:
            puzzle = puzzle * (self.grid_size ** 2)
            solution = solution * (self.grid_size ** 2)

        size = self.grid_size ** 2

        print(f"Sample {idx}: Sudoku Puzzle")

        print("\nPUZZLE (0 = empty):")
        self._print_grid(puzzle.numpy(), size)

        print("\nSOLUTION:")
        self._print_grid(solution.numpy(), size)

    def _print_grid(self, flat_grid, size):
        """Print a flattened grid in 2D format"""
        for i in range(size):
            row = flat_grid[i*size:(i+1)*size]
            row_str = ' '.join(
                [f'{int(x):2d}' if x > 0 else ' .' for x in row])

            # Add horizontal dividers for 9×9
            if size == 9 and i % 3 == 0 and i > 0:
                print('  ' + '-' * (size * 3))

            # Add vertical dividers for 9×9
            if size == 9:
                parts = []
                for j in range(0, size*3, 9):
                    parts.append(row_str[j:j+9])
                print('  ' + ' | '.join(parts))
            else:
                print('  ' + row_str)


if __name__ == "__main__":

    print("TESTING SUDOKU DATASET")

    # Test 1: Small dataset (4×4 Sudoku)
    print("\n[TEST 1] Generating 4×4 Sudoku (fast test)")
    small_dataset = SudokuDataset(
        num_samples=10,
        grid_size=2,  # 2 = 4×4 grid
        difficulty=0.3,
        seed=42,
        normalize=True
    )
    small_dataset.visualize_sample(0)

    # Test 2: Paper-aligned dataset (9×9 Sudoku, 1000 samples)
    print("\n[TEST 2] Generating 9×9 Sudoku (paper configuration)")
    train_dataset = SudokuDataset(
        num_samples=100,  # Use 100 for quick test (paper uses 1000)
        grid_size=3,  # 3 = 9×9 grid
        difficulty=0.5,
        seed=42,
        normalize=True
    )
    train_dataset.visualize_sample(0)

    # Test 3: DataLoader compatibility
    print("\n[TEST 3] Testing PyTorch DataLoader compatibility")
    from torch.utils.data import DataLoader

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    batch_puzzles, batch_solutions = next(iter(loader))

    print(f" Batch puzzles shape: {batch_puzzles.shape}")
    print(f" Batch solutions shape: {batch_solutions.shape}")
    print(
        f" Puzzle value range: [{batch_puzzles.min():.2f}, {batch_puzzles.max():.2f}]")
    print(
        f" Solution value range: [{batch_solutions.min():.2f}, {batch_solutions.max():.2f}]")

    # Test 4: Different difficulties
    print("\n[TEST 4] Testing different difficulty levels")
    for diff in [0.3, 0.5, 0.7]:
        test_ds = SudokuDataset(
            num_samples=5, grid_size=3, difficulty=diff, seed=42)
        empty_ratio = (test_ds.puzzles[0] == 0).sum(
        ).item() / len(test_ds.puzzles[0])
        print(f"  Difficulty {diff:.1f} → {empty_ratio:.1%} empty cells")

    print(" ALL TESTS PASSED")
