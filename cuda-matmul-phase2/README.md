# CUDA Matrix Multiplication - Phase 2: Mixed Precision

Mixed-precision matrix multiplication using FP16 storage with FP32 accumulation, benchmarked on Tesla P100.

![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)
![Performance](https://img.shields.io/badge/Performance-6.4_TFLOPS-success)
![GPU](https://img.shields.io/badge/GPU-Tesla_P100-blue)

## Key Results

```
┌─────────────────────────────────────────────────────────────┐
│  Peak Performance: 4,731 GFLOPS (Mixed + Register Blocked)  │
│  Speedup vs FP32:  2.6× on 2048³ matrices                   │
│  GPU: Tesla P100-PCIE-16GB                                  │
└─────────────────────────────────────────────────────────────┘
```

[**Phase 2 Technical**](docs/PHASE2_MIXED_PRECISION.md) | Deep dive into FP16/FP32 precision


## The Core Insight

```
FP16 storage  → 2× memory bandwidth (2 bytes vs 4 bytes)
FP32 accumulate → Maintains numerical accuracy
= Best of both worlds!
```

## Benchmark Results (Tesla P100)

### Performance Comparison

| Matrix Size | Method                     | Time (ms) | GFLOPS | vs FP32 |
|-------------|----------------------------|-----------|--------|---------|
| **1024³**   | FP32 (baseline)            | 1.225     | 1,753  | 1.00×   |
|             | FP16 Pure                  | 1.190     | 1,805  | 1.03×   |
|             | Mixed Precision            | 1.446     | 1,485  | 0.85×   |
|             | **Mixed + Register Blocked** | **0.580** | **3,701** | **2.11×** |
| **2048³**   | FP32 (baseline)            | 9.455     | 1,817  | 1.00×   |
|             | FP16 Pure                  | 8.899     | 1,931  | 1.06×   |
|             | Mixed Precision            | 10.174    | 1,689  | 0.93×   |
|             | **Mixed + Register Blocked** | **3.631** | **4,731** | **2.60×** |

### Numerical Accuracy (1024³, scale=1.0)

| Method             | RMS Error  | Max Abs Error | Notes                     |
|--------------------|------------|---------------|---------------------------|
| FP32 Baseline      | 2.45e-6    | 1.91e-5       | Reference accuracy        |
| FP16 Pure          | 5.02e-2    | 6.35e-1       | **20,000× worse**         |
| Mixed Precision    | 2.79e-3    | 1.34e-2       | Input quantization loss   |

### Overflow Demonstration (scale=100)

```
FP16 Pure Results:
  OVERFLOW DETECTED: 986,967 inf values (94% of matrix!)
  
Mixed Precision Results:
  No overflow - FP32 accumulation handles large values
```

This demonstrates why pure FP16 fails in practice - values exceeding 65,504 become infinity.

## Why Register Blocking Matters

Basic mixed precision is actually **slower** than FP32 due to conversion overhead:

```
FP32 baseline:     1.225 ms
Mixed (basic):     1.446 ms  ← 18% SLOWER (conversion overhead)
Mixed + RegBlock:  0.580 ms  ← 2.1× FASTER (amortized conversions)
```

Register blocking does **16× more work per thread**, amortizing the FP16↔FP32 conversion cost.

## Connection to Floating-Point Forensics

This phase applies concepts from the FP Forensics Toolkit:

| FP Forensics Concept        | Observed in Benchmarks                              |
|-----------------------------|-----------------------------------------------------|
| FP16: 10-bit mantissa       | Mixed precision ~1000× worse than pure FP32         |
| FP16 max: 65,504            | scale=100 caused 94% overflow                       |
| Accumulation errors         | FP16 pure: 20,000× worse RMS error                  |
| FP32 accumulation           | Mixed precision prevented all overflow              |

### IEEE 754 Format Comparison

```
FP16:  [1 sign] [5 exponent] [10 mantissa]  = 16 bits (~3 decimal digits)
FP32:  [1 sign] [8 exponent] [23 mantissa]  = 32 bits (~7 decimal digits)
```

## Files

```
phase2-mixed-precision/
├── matmul_mixed_precision.cu    # CUDA implementation (5 kernel variants)
├── test_mixed_precision.py      # Test suite
├── PHASE2_MIXED_PRECISION.md    # Detailed documentation
└── README.md                    # This file
```

## Quick Start

### Compile

```bash
# Tesla P100 (sm_60), V100 (sm_70), A100 (sm_80)
nvcc -O3 -arch=sm_60 -o matmul_mixed matmul_mixed_precision.cu
```

### Run Benchmarks

```bash
./matmul_mixed 1024 1024 1024           # Normal test
./matmul_mixed 2048 2048 2048           # Larger (better GPU utilization)
./matmul_mixed 1024 1024 1024 100       # Trigger FP16 overflow
./matmul_mixed 1024 1024 1024 0.001     # Small values test
```

## Kernel Variants

| Kernel                                      | Input | Accumulator | Output | Use Case                     |
|---------------------------------------------|-------|-------------|--------|------------------------------|
| `matmul_fp32`                               | FP32  | FP32        | FP32   | Baseline                     |
| `matmul_fp16_pure`                          | FP16  | FP16        | FP16   | Educational (shows problems) |
| `matmul_mixed_precision`                    | FP16  | FP32        | FP32   | Basic mixed                  |
| `matmul_mixed_precision_register_blocked`   | FP16  | FP32        | FP32   | **Best performance**         |
| `matmul_mixed_precision_fp16_output`        | FP16  | FP32        | FP16   | Chained operations           |

## Real-World Applications

This is the same strategy used by:
- **PyTorch AMP** (Automatic Mixed Precision)
- **TensorFlow** mixed precision training
- **NVIDIA Tensor Cores** (hardware FP16×FP16→FP32)
- All modern deep learning frameworks

## Project Progression

```
Phase 1: Rectangular Matrices     Complete
Phase 2: Mixed Precision          Complete (this phase)
Phase 3: Batched GEMM             → Next
```

## Hardware

- **GPU**: NVIDIA Tesla P100-PCIE-16GB
- **Architecture**: Pascal (sm_60)
- **CUDA Cores**: 3,584
- **Memory**: 16GB HBM2
- **Theoretical Peak**: 9,300 GFLOPS (FP32)

## Author

**Andrey Maltsev**  
GPU Computing & High-Performance Computing

## License

MIT License
