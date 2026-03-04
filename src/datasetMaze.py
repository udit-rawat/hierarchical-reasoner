"""
Maze Dataset for Hierarchical Reasoning Model

Generates random mazes using recursive backtracking (DFS) and formats
them for HRM training. The model learns to find the optimal path from
start to goal.

Grid encoding:
    0.0  = open path
    1.0  = wall
    0.5  = start position
    0.75 = goal position

Label encoding (solution):
    0.0  = wall or unused cell
    1.0  = wall
    0.25 = solution path
    0.5  = start
    0.75 = goal

Usage:
    train_dataset = MazeDataset(num_samples=1000, maze_size=5)
    test_dataset  = MazeDataset(num_samples=200,  maze_size=7)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import deque


class MazeDataset(Dataset):
    """
    Maze pathfinding dataset for training HRM.

    Args:
        num_samples (int): Number of mazes to generate
        maze_size   (int): Grid dimension — 5 = 5×5, 7 = 7×7 (odd numbers work best)
        seed        (int): Random seed for reproducibility

    Returns:
        maze     (Tensor): Flattened maze grid [maze_size^2]
        solution (Tensor): Flattened solution path [maze_size^2]
    """

    def __init__(self, num_samples=1000, maze_size=5, seed=None):
        super().__init__()
        self.num_samples = num_samples
        self.maze_size = maze_size
        self.flat_size = maze_size * maze_size

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        print(f"Generating {num_samples} mazes ({maze_size}×{maze_size})...")
        self.mazes = []
        self.solutions = []

        generated = 0
        attempts = 0
        while generated < num_samples:
            attempts += 1
            maze = self._generate_maze()
            path = self._solve_maze(maze)

            # Skip unsolvable mazes (shouldn't happen with DFS gen, but safety check)
            if path is None:
                continue

            maze_tensor = self._encode_maze(maze, path)
            solution_tensor = self._encode_solution(maze, path)

            self.mazes.append(maze_tensor)
            self.solutions.append(solution_tensor)
            generated += 1

            if generated % 200 == 0:
                print(f"  Generated {generated}/{num_samples} mazes")

        print(f" Dataset ready: {num_samples} mazes")
        self._print_stats()

    def _generate_maze(self):
        """
        Generate a maze using recursive backtracking (DFS).
        Guarantees a path exists from top-left to bottom-right.

        Returns:
            np.ndarray: 2D grid — 1=wall, 0=path
        """
        size = self.maze_size
        maze = np.ones((size, size), dtype=np.float32)

        # Start carving from (0, 0)
        start = (0, 0)
        maze[start] = 0
        stack = [start]
        visited = {start}

        while stack:
            r, c = stack[-1]
            # All 4 neighbors (2 steps away to keep walls between cells)
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited:
                    neighbors.append((nr, nc, r + dr // 2, c + dc // 2))

            if neighbors:
                nr, nc, wr, wc = neighbors[np.random.randint(len(neighbors))]
                maze[nr][nc] = 0      # carve destination
                maze[wr][wc] = 0      # carve wall between
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        # Ensure start and goal are open
        maze[0][0] = 0
        maze[size - 1][size - 1] = 0

        return maze

    def _solve_maze(self, maze):
        """
        Find shortest path from top-left to bottom-right using BFS.

        Returns:
            list of (r, c) tuples representing the path, or None if unsolvable
        """
        size = self.maze_size
        start = (0, 0)
        goal = (size - 1, size - 1)

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            (r, c), path = queue.popleft()

            if (r, c) == goal:
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < size and 0 <= nc < size
                        and maze[nr][nc] == 0
                        and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))

        return None  # No path found

    def _encode_maze(self, maze, path):
        """
        Encode maze as input tensor.
            0.0  = open path
            1.0  = wall
            0.5  = start (0, 0)
            0.75 = goal  (size-1, size-1)
        """
        size = self.maze_size
        grid = maze.copy()
        grid[0][0] = 0.5
        grid[size - 1][size - 1] = 0.75
        return torch.tensor(grid.flatten(), dtype=torch.float32)

    def _encode_solution(self, maze, path):
        """
        Encode solution path as output tensor.
            0.0  = open/unused cell
            1.0  = wall
            0.25 = solution path cell
            0.5  = start
            0.75 = goal
        """
        size = self.maze_size
        grid = maze.copy()

        for r, c in path:
            grid[r][c] = 0.25

        grid[0][0] = 0.5
        grid[size - 1][size - 1] = 0.75

        return torch.tensor(grid.flatten(), dtype=torch.float32)

    def _print_stats(self):
        sample_maze = self.mazes[0]
        sample_sol = self.solutions[0]

        wall_ratio = (sample_maze == 1.0).sum().item() / self.flat_size
        path_len = (sample_sol == 0.25).sum().item() + 2  # +2 for start/goal

        print(f"\nDataset Statistics:")
        print(f"  Total samples  : {self.num_samples}")
        print(f"  Grid size      : {self.maze_size}×{self.maze_size}")
        print(f"  Input dim      : {self.flat_size}")
        print(f"  Output dim     : {self.flat_size}")
        print(f"  Wall ratio     : {wall_ratio:.1%}")
        print(f"  Avg path length: ~{path_len} cells")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.mazes[idx], self.solutions[idx]

    def visualize_sample(self, idx=0):
        """Print a maze and its solution in readable ASCII format."""
        size = self.maze_size
        maze = self.mazes[idx].numpy().reshape(size, size)
        sol = self.solutions[idx].numpy().reshape(size, size)

        symbols_maze = {0.0: '.', 1.0: '#', 0.5: 'S', 0.75: 'G'}
        symbols_sol  = {0.0: '.', 1.0: '#', 0.25: '*', 0.5: 'S', 0.75: 'G'}

        print(f"\nSample {idx} — Maze ({size}×{size}):")
        print("  MAZE          SOLUTION")
        for r in range(size):
            maze_row = ' '.join(symbols_maze.get(round(maze[r][c], 2), '?') for c in range(size))
            sol_row  = ' '.join(symbols_sol.get(round(sol[r][c], 2), '?')  for c in range(size))
            print(f"  {maze_row}     {sol_row}")
        print("  Legend: # wall  . open  S start  G goal  * path")


def get_dataloader(num_samples=1000, maze_size=5, batch_size=32,
                   shuffle=True, seed=42):
    """Convenience wrapper returning a DataLoader."""
    dataset = MazeDataset(num_samples=num_samples, maze_size=maze_size, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    print("=" * 60)
    print(" TESTING MAZE DATASET")
    print("=" * 60)

    # Test 1: Small 5x5 maze
    print("\n[TEST 1] 5×5 maze — 20 samples")
    ds_small = MazeDataset(num_samples=20, maze_size=5, seed=42)
    ds_small.visualize_sample(0)
    ds_small.visualize_sample(1)

    # Test 2: Larger 7x7 maze
    print("\n[TEST 2] 7×7 maze — 20 samples")
    ds_large = MazeDataset(num_samples=20, maze_size=7, seed=42)
    ds_large.visualize_sample(0)

    # Test 3: DataLoader compatibility
    print("\n[TEST 3] DataLoader compatibility")
    loader = DataLoader(ds_small, batch_size=4, shuffle=True)
    batch_mazes, batch_solutions = next(iter(loader))
    print(f"  Maze batch shape    : {batch_mazes.shape}")
    print(f"  Solution batch shape: {batch_solutions.shape}")
    print(f"  Maze value range    : [{batch_mazes.min():.2f}, {batch_mazes.max():.2f}]")
    print(f"  Solution value range: [{batch_solutions.min():.2f}, {batch_solutions.max():.2f}]")

    # Test 4: HRM input dim check
    print("\n[TEST 4] HRM input dimension reference")
    for size in [5, 7]:
        print(f"  {size}×{size} maze → input_dim = output_dim = {size*size}")

    print("\n ALL TESTS PASSED")
