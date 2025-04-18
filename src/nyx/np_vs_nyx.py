import time
import numpy as np
from src.array import Array

def benchmark(fn, *args):
    start = time.time()
    result = fn(*args)
    return result, time.time() - start

def test_performance():
    # Przygotuj dane
    size = 1000
    a_np = np.random.rand(size)
    b_np = np.random.rand(size)
    a_nyx = Array(list(a_np))
    b_nyx = Array(list(b_np))

    print("Benchmark dla 1D dot():")
    _, t1 = benchmark(np.dot, a_np, b_np)
    _, t2 = benchmark(a_nyx.dot, b_nyx)
    print(f"NumPy: {t1:.6f}s | Nyx: {t2:.6f}s")

    print("\nBenchmark dla sum():")
    _, t1 = benchmark(np.sum, a_np)
    _, t2 = benchmark(a_nyx.sum)
    print(f"NumPy: {t1:.6f}s | Nyx: {t2:.6f}s")

    print("\nBenchmark dla mean():")
    _, t1 = benchmark(np.mean, a_np)
    _, t2 = benchmark(a_nyx.mean)
    print(f"NumPy: {t1:.6f}s | Nyx: {t2:.6f}s")

    print("\nBenchmark dla matmul():")
    mat_a_np = np.random.rand(100, 100)
    mat_b_np = np.random.rand(100, 100)
    mat_a_nyx = Array(mat_a_np.tolist())
    mat_b_nyx = Array(mat_b_np.tolist())
    _, t1 = benchmark(np.matmul, mat_a_np, mat_b_np)
    _, t2 = benchmark(mat_a_nyx.matmul, mat_b_nyx)
    print(f"NumPy: {t1:.6f}s | Nyx: {t2:.6f}s")

if __name__ == "__main__":
    test_performance()
