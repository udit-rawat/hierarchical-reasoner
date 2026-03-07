"""
Experiment: Inference Speed Benchmark — HRM vs RNN
Measures ms/sample across batch sizes on 4x4 Sudoku input size.

Usage:
    python3 experiments/benchmark_inference.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torch.nn as nn
from src.model import HierarchicalReasoningModel
from experiments.run_baseline_rnn import RNNBaseline

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
WARMUP = 50
RUNS = 500
INPUT_DIM = 16   # 4x4 Sudoku
HIDDEN_DIM = 64
SEQ_LEN = 1

configs = [
    ("RNN",        lambda: RNNBaseline(INPUT_DIM, HIDDEN_DIM, INPUT_DIM)),
    ("HRM (1-step)", lambda: HierarchicalReasoningModel(INPUT_DIM, HIDDEN_DIM, INPUT_DIM, num_steps=1, l_iterations=5)),
    ("HRM (3-step)", lambda: HierarchicalReasoningModel(INPUT_DIM, HIDDEN_DIM, INPUT_DIM, num_steps=3, l_iterations=5)),
    ("HRM (5-step)", lambda: HierarchicalReasoningModel(INPUT_DIM, HIDDEN_DIM, INPUT_DIM, num_steps=5, l_iterations=5)),
]

BATCH_SIZES = [1, 8, 32, 128]


def benchmark(model, batch_size):
    model.eval()
    x = torch.randn(batch_size, SEQ_LEN, INPUT_DIM).to(DEVICE)

    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)

    if DEVICE.type == 'mps':
        torch.mps.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(RUNS):
            model(x)
    if DEVICE.type == 'mps':
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_batch = (elapsed / RUNS) * 1000
    ms_per_sample = ms_per_batch / batch_size
    return ms_per_batch, ms_per_sample


def main():
    print("=" * 72)
    print(f"  Inference Speed Benchmark — HRM vs RNN  |  device: {DEVICE}")
    print(f"  {RUNS} runs, {WARMUP} warmup, input_dim={INPUT_DIM}, hidden={HIDDEN_DIM}")
    print("=" * 72)

    for batch_size in BATCH_SIZES:
        print(f"\n  Batch size: {batch_size}")
        print(f"  {'Model':<18}  {'Params':>8}  {'ms/batch':>9}  {'ms/sample':>10}  {'vs RNN':>8}")
        print("  " + "-" * 60)

        rnn_ms = None
        for name, build_fn in configs:
            model = build_fn().to(DEVICE)
            params = sum(p.numel() for p in model.parameters())
            ms_batch, ms_sample = benchmark(model, batch_size)

            if rnn_ms is None:
                rnn_ms = ms_sample
                speedup = "—"
            else:
                ratio = ms_sample / rnn_ms
                speedup = f"{ratio:.2f}x"

            print(f"  {name:<18}  {params:>8,}  {ms_batch:>8.3f}  {ms_sample:>9.4f}  {speedup:>8}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
