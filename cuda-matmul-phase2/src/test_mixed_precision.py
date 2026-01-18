#!/usr/bin/env python3
"""
Test Suite for Mixed Precision Matrix Multiplication

Tests numerical accuracy and edge cases for mixed precision computation.
Connects to Floating-Point Forensics concepts.
"""

import numpy as np
import sys

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("  CuPy not available - running NumPy-only tests")

sys.path.insert(0, 'src')    
from matmul_mixed_precision import (
    MixedPrecisionMatMul,
    compute_precision_stats,
    analyze_fp16_representation,
    demonstrate_accumulation_error
)


def test_fp16_representation():
    """Test FP16 representation accuracy."""
    print("\n" + "="*60)
    print("FP16 REPRESENTATION TESTS")
    print("="*60)

    test_cases = [
        (0.1, "Common decimal"),
        (0.5, "Exact binary"),
        (1.0, "Unity"),
        (1000.0, "Large integer"),
        (0.0001, "Small value"),
        (65504.0, "Max FP16"),
        (65505.0, "Beyond max FP16"),
    ]

    all_passed = True

    for value, desc in test_cases:
        result = analyze_fp16_representation(value)

        if value <= 65504:
            is_inf = np.isinf(np.float16(value))
            if is_inf:
                print(f"x {desc} ({value}): Unexpected overflow")
                all_passed = False
            else:
                rel_err = result['relative_error']
                passed = rel_err < 0.01 or value < 1e-4
                status = "OK" if passed else "NOT  OK"
                print(f"{status} {desc} ({value}): rel_err={rel_err:.2e}")
                if not passed:
                    all_passed = False
        else:
            is_inf = np.isinf(np.float16(value))
            status = "OK" if is_inf else "NOT OK"
            print(f"{status} {desc} ({value}): overlow={'yes' if is_inf else 'no'}")
            if not is_inf:
                all_passed = False

    return all_passed


def test_accumulation_error():
    """Test that FP16 accumulation has more error than FP32."""
    print("\n" + "="*60)
    print("ACCUMULATION ERROR TESTS")
    print("="*60)

    all_passed = True

    for n in [100, 1000, 10000]:
        result = demonstrate_accumulation_error(n, 0.1)

        error_ratio = result['error_ratio']
        passed = error_ratio > 10 or n < 100
        status = "OK" if passed else "NOT OK"

        print(f"{status} n={n}: FP16 error is {error_ratio:.0f}* worse than FP32")

        if not passed:
            all_passed = False

    return all_passed


def test_mixed_precision_accuracy():
    """Test that mixed precision maintains FP32-like accuracy."""
    print("\n" + "="*60)
    print("MIXED PRECISION ACCURACY TESTS")
    print("="*60)

    if not HAS_CUPY:
        print("Skipping - CuPy required")
        return True

    matmul = MixedPrecisionMatMul()
    all_passed = True

    test_cases = [
        (256, 256, 256, 1.0, "Small square"),
        (512, 256, 128, 1.0, "Rectangular"),
        (256, 256, 256, 0.1, "Small values"),
        (256, 256, 256, 10.0, "Larger values"), 
    ]

    for M, K, N, scale, desc in test_cases:
        np.random.seed(42)
        A_np = (np.random.randn(M,K) * scale).astype(np.float32)
        B_np = (np.random.randn(K, N) * scale).astype(np.float32)

        C_ref = np.matmul(A_np, B_np)

        A_gpu = cp.array(A_np)
        B_gpu = cp.array(B_np)

        C_fp32 = cp.astype(matmul.multiply_fp32(A_gpu, B_gpu))
        stats_fp32 = compute_precision_stats(C_fp32, C_ref)

        C_mixed = cp.asnumpy(matmul.multiply_mixed(A_gpu, B_gpu))
        stats_mixed = compute_precision_stats(C_mixed, C_ref)

        C_fp16 = cp.asnumpy(matmul.multiply_fp16_pure(A_gpu, B_gpu))
        stats_fp16 = compute_precision_stats(C_fp16, C_ref)

        mixed_vs_fp32 = stats_mixed.max_rel_error / (stats_fp32.max_rel_error + 1e-10)
        fp16_vs_mixed = stats_fp16.max_rel_error / (stats_mixed.max_rel_error + 1e-10)

        passed = mixed_vs_fp32 < 100 and fp16_vs_mixed > 1
        status = "OK" if passed else "NOT OK"

        print(f"{status} {desc}:")
        print(f"      FP32:  {stats_fp32.max_rel_error:.2e}")
        print(f"      Mixed: {stats_mixed.max_rel_error:.2e} ({mixed_vs_fp32:.1f}× vs FP32)")
        print(f"      FP16:  {stats_fp16.max_rel_error:.2e} ({fp16_vs_mixed:.1f}× vs Mixed)")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_overflow_detection():
    """Test that FP16 overflow is detected correctly."""
    print("\n" + "="*60)
    print("OVERFLOW DETECTION TESTS")
    print("="*60)

    if not HAS_CUPY:
        print("Skipping - CuPy required")
        return True
    
    matmul = MixedPrecisionMatMul()
    all_passed = True

    np.random.seed(42)
    scale = 500
    M, K, N = 128, 128, 128

    A_np = (np.random.randn(M, K) * scale).astype(np.float32)
    B_np = (np.random.randn(K, N) * scale).astype(np.float32)

    A_gpu = cp.array(A_np)
    B_gpu = cp.array(B_np)

    C_fp16 = cp.asnumpy(matmul.multiply_fp16_pure(A_gpu, B_gpu))
    stats_fp16 = compute_precision_stats(C_fp16, np.zeros_like(C_fp16))

    has_overflow = stats_fp16.int_count > 0 or stats_fp16.nan_count > 0

    C_mixed = cp.asnumpy(matmul.multiply_mixed(A_gpu, B_gpu))
    stats_mixed = compute_precision_stats(C_mixed, np.zeros_like(C_mixed))

    mixed_ok = stats_mixed.inf_count == 0 and stats_mixed.nan_count == 0

    print(f" Scale={scale}, expecting FP16 overflow")
    print(f"  FP16 Pure: {stats_fp16.inf_count} inf, {stats_fp16.nan_count} nan")
    print(f"  Mixed:     {stats_mixed.inf_count} inf, {stats_mixed.nan_count} nan")

    passed = has_overflow and mixed_ok
    status = "OK" if passed else "NOT OK"
    print(f"\n{status} FP16 overflow detected: {has_overflow}, Mixed stable: {mixed_ok}")

    if not passed:
        all_passed = False

    return all_passed


def test_edge_cases():
    """Test edge cases for mixed precision."""
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)
    
    if not HAS_CUPY:
        print("Skipping - CuPy required")
        return True
    
    matmul = MixedPrecisionMatMul()
    all_passed = True

    edge_cases = [
        (1, 1, 1, "Minimum size"),
        (17, 17, 17, "Non-aligned"),
        (64, 1, 64, "K=1 (outer product)"),
        (1, 64, 1, "M=N=1"),
        (128, 128, 128, "Normal aligned"),
    ]

    for M, K, N, desc in edge_cases:
        np.random.seed(42)
        A_np = np random.randn(M, K).astype(np.float32) 
        B_np = np.random.randn(K, N).astype(np.float32)
        C_ref = np.matmul(A_np, B_np)

        A_gpu = cp.array(A_np)
        B_gpu = cp. array(B_np)

        try:
            C_mixed = cp.asnumpy(matmul.multiply_mixed(A_gpu, B_gpu))
            stats = compute_precision_stats(C_mixed, C_ref)

            passed = stats.max_rel_error < 0.01
            status = "OK" if passed else "NOT OK"
            print(f"{status} {desc} ({M}×{K}×{N}): rel_err={stats.max_rel_error:.2e}")

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"x {desc} ({M}x{K}x{N}) : EXCEPTION - {e}")
            all_passed = False

    return all_passed


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("MIXED PRECISION TEST SUITE")
    print("="*60)
    
    results = {
        'fp16_representation': test_fp16_representation(),
        'accumulation_error': test_accumulation_error(),
        'mixed_accuracy': test_mixed_precision_accuracy(),
        'overflow_detection': test_overflow_detection(),
        'edge_cases': test_edge_cases(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}" )
        if not passed:
            all_passed = False

    print("\n" + ("="*60))
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60 + "\n")

    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
        