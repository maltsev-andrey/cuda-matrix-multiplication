#!/usr/bin/env python3
"""
Test Suite for Batched GEMM

Tests correctness and performance scaling for batched matrix multiplication.
"""

import numpy as np
import subprocess
import sys


def run_benchmark(M, K, N, batch_size):
    """Run the CUDA benchmark and parse results."""
    try:
        result = subprocess.run(
            ['./matmul_batched', str(M), str(K), str(N), str(batch_size)],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout
    except FileNotFoundError:
        print("Error: matmul_batched not found. Compile first with:")
        print("  nvcc -O3 -arch=sm_60 -o matmul_batched matmul_batched.cu")
        return None
    except subprocess.TimeoutExpired:
        print(f"Timeout for {M}×{K}×{N}, batch={batch_size}")
        return None


def test_various_configurations():
    """Test different batch sizes and matrix dimensions."""
    print("\n" + "="*60)
    print("BATCHED GEMM TEST CONFIGURATIONS")
    print("="*60)
    
    configs = [
        # (M, K, N, batch_size, description)
        (64, 64, 64, 8, "Transformer attention heads"),
        (128, 128, 128, 32, "Small matrices, large batch"),
        (256, 256, 256, 16, "Medium matrices"),
        (512, 512, 512, 4, "Larger matrices, small batch"),
        (64, 768, 512, 8, "Non-square (embedding layer)"),
        (128, 64, 128, 12, "Attention-like Q×K^T"),
    ]
    
    for M, K, N, batch, desc in configs:
        print(f"\n{'─'*60}")
        print(f"Test: {desc}")
        print(f"Config: {batch} × ({M}×{K}) × ({K}×{N})")
        print(f"{'─'*60}")
        
        output = run_benchmark(M, K, N, batch)
        if output:
            # Extract key performance lines
            for line in output.split('\n'):
                if 'GFLOPS' in line or 'Speedup' in line or 'launches' in line:
                    print(line)


def test_batch_scaling():
    """Test how performance scales with batch size."""
    print("\n" + "="*60)
    print("BATCH SIZE SCALING TEST")
    print("="*60)
    
    M, K, N = 128, 128, 128
    batch_sizes = [4, 8, 16, 32, 64]
    
    print(f"\nMatrix size: {M}×{K}×{N}")
    print(f"Testing batch sizes: {batch_sizes}")
    print()
    
    for batch in batch_sizes:
        output = run_benchmark(M, K, N, batch)
        if output:
            for line in output.split('\n'):
                if 'Batched + Mixed + RegBlock' in line:
                    print(f"Batch={batch:3d}: {line.strip()}")


def verify_correctness_cpu():
    """Simple CPU verification test."""
    print("\n" + "="*60)
    print("CPU CORRECTNESS VERIFICATION")
    print("="*60)
    
    M, K, N = 64, 64, 64
    batch_size = 4
    
    print(f"\nGenerating {batch_size} random {M}×{K} and {K}×{N} matrices...")
    
    np.random.seed(42)
    
    # Generate batched matrices
    A = np.random.randn(batch_size, M, K).astype(np.float32)
    B = np.random.randn(batch_size, K, N).astype(np.float32)
    
    # Compute reference
    C_ref = np.zeros((batch_size, M, N), dtype=np.float32)
    for b in range(batch_size):
        C_ref[b] = np.matmul(A[b], B[b])
    
    print(f"Reference computation complete.")
    print(f"\nSample outputs:")
    for b in range(min(2, batch_size)):
        print(f"  C[{b}][0,0] = {C_ref[b, 0, 0]:.6f}")
    
    print("\n CPU reference computed successfully")
    print("  (GPU verification happens in CUDA program)")


def main():
    print("\n" + "="*60)
    print("BATCHED GEMM TEST SUITE")
    print("="*60)
    
    # Check if binary exists
    import os
    if not os.path.exists('./matmul_batched'):
        print("\n matmul_batched binary not found!")
        print("Compile first:")
        print("  nvcc -O3 -arch=sm_60 -o matmul_batched matmul_batched.cu")
        print("\nRunning CPU verification only...")
        verify_correctness_cpu()
        return
    
    verify_correctness_cpu()
    test_various_configurations()
    test_batch_scaling()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

