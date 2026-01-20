# CUDA Matrix Multiplication Optimization

High-performance GPU-accelerated matrix multiplication achieving **6,590 GFLOPS** (71% of theoretical peak) on NVIDIA Tesla P100 through progressive CUDA optimization.

![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)
![Performance](https://img.shields.io/badge/Performance-6.6_TFLOPS-success)
![GPU](https://img.shields.io/badge/GPU-Tesla_P100-blue)

---

## Project Roadmap

```
+----------------------------------------------------------------------------+
|                    CUDA Matrix Multiplication Phases                       |
+----------------------------------------------------------------------------+
|                                                                            |
| OK Base Implementation          Shared memory tiling, register blocking    |
|    +-- 6,436 GFLOPS (69% peak)  92% of cuBLAS efficiency                   |
|                                                                            |
| OK Phase 1: Rectangular         Support for M!=N!=K matrices               |
|    +-- Integrated in base       Real-world matrix shapes                   |
|                                                                            |
| OK Phase 2: Mixed Precision     FP16 storage, FP32 accumulation            |
|    +-- 4,731 GFLOPS             2.6x speedup, 2x bandwidth                 |
|                                                                            |
| OK Phase 3: Batched GEMM        Multiple matrices per kernel               |
|    +-- 6,590 GFLOPS (71% peak)  Up to 5.75x speedup vs naive loop          |
|                                                                            |
|  o Phase 4: Tensor Cores        Volta+ GPU acceleration                    |
|    +-- Future                   Potential 8x speedup                       |
|                                                                            |
|  o Phase 5: Multi-GPU           NCCL distributed computing                 |
|    +-- Future                   Scale beyond single GPU                    |
|                                                                            |
|  Legend: OK Complete  o Planned                                            |
+----------------------------------------------------------------------------+
```

---

## Phase Overview

| Phase                                                    | Status    | Performance   | Description                       |
|----------------------------------------------------------|-----------|---------------|-----------------------------------|
| [**Base Implementation**](#base-implementation)          | Complete  | 6,436 GFLOPS  | Shared memory + register blocking |
| [**Phase 1: Rectangular**](#phase-1-rectangular-matrices)| Complete  | Integrated    | Support for M!=N!=K               |
| [**Phase 2: Mixed Precision**](cuda-matmul-phase2/README.md) | Complete  | 4,731 GFLOPS  | FP16/FP32 mixed precision         |
| [**Phase 3: Batched GEMM**](batched_gemm_phase3/README.md)| Complete  | 6,590 GFLOPS  | Multiple matrix pairs per kernel  |
| [Phase 4: Tensor Cores](#phase-4-tensor-cores-future)    | o Future  | -             | Hardware FP16 acceleration        |
| [Phase 5: Multi-GPU](#phase-5-multi-gpu-future)          | o Future  | -             | Distributed computing             |

---

## Base Implementation

### Key Achievements

- **6,436 GFLOPS** peak performance on Tesla P100
- **69% of theoretical peak** (9,300 GFLOPS)
- **92% of cuBLAS efficiency** (NVIDIA's optimized library)
- **379% performance improvement** through progressive optimization

### Optimization Levels

| Level | Technique                     | Performance  | Improvement |
|-------|-------------------------------|--------------|-------------|
| 1     | Shared Memory Tiling (16x16)  | 1,732 GFLOPS | Baseline    |
| 2     | Larger Tiles (32x32)          | 2,080 GFLOPS | +55%        |
| 3     | Register Blocking (4x4)       | 6,436 GFLOPS | **+379%**   |

### Source Files

| File                                      | Description                                |
|-------------------------------------------|--------------------------------------------|
| [`matmul_kernels.cu`](matmul_kernels.cu)  | Register-blocked implementation (best performance) |
| [`matmul_kernels_shared_memory.cu`](matmul_kernels_shared_memory.cu) | Shared memory baseline |

### Quick Start

```bash
# Compile
nvcc -O3 -arch=sm_60 -o matmul_register matmul_kernels.cu

# Run
./matmul_register 1024 1024 1024      # Small test
./matmul_register 16384 16384 16384   # Peak performance
```

---

## Phase 1: Rectangular Matrices

**Status**: Complete (integrated in base implementation)

Support for non-square matrices: `C(MxN) = A(MxK) x B(KxN)`

Already supported in `matmul_kernels.cu` with proper boundary handling for M!=N!=K.

```bash
./matmul_register 2048 512 4096    # Rectangular matrices
```

---

## Phase 2: Mixed Precision

**Status**: Complete | **Performance**: 4,731 GFLOPS | **Speedup**: 2.6x vs FP32

**[Full Documentation](cuda-matmul-phase2/README.md)**

### Key Results

```
+-------------------------------------------------------------+
|  FP16 storage  -> 2x memory bandwidth                       |
|  FP32 accumulate -> Maintains numerical accuracy            |
|  = Best of both worlds!                                     |
+-------------------------------------------------------------+
```

| Matrix   | Method                        | GFLOPS  | vs FP32 |
|----------|-------------------------------|---------|---------|
| 2048^3   | FP32 Baseline                 | 1,817   | 1.00x   |
| 2048^3   | **Mixed + Register Blocked**  | **4,731** | **2.60x** |

### Overflow Demonstration

```
FP16 Pure (scale=100):
  OVERFLOW: 986,967 inf values (94% of matrix!)
  
Mixed Precision:
  No overflow - FP32 accumulation handles large values
```

### Source Files

| File                                                                 | Description            |
|----------------------------------------------------------------------|------------------------|
| [`cuda-matmul-phase2/src/matmul_mixed_precision.cu`](cuda-matmul-phase2/src/matmul_mixed_precision.cu) | Mixed precision kernels |
| [`cuda-matmul-phase2/src/test_mixed_precision.py`](cuda-matmul-phase2/src/test_mixed_precision.py)     | Test suite             |

### Quick Start

```bash
cd cuda-matmul-phase2/src
nvcc -O3 -arch=sm_60 -o matmul_mixed matmul_mixed_precision.cu
./matmul_mixed 2048 2048 2048           # Normal benchmark
./matmul_mixed 1024 1024 1024 100       # Trigger FP16 overflow
```

---

## Phase 3: Batched GEMM

**Status**: Complete | **Performance**: 6,590 GFLOPS (71% peak) | **Speedup**: Up to 5.75x

**[Full Documentation](batched_gemm_phase3/README.md)**

### Key Concept

Multiple matrix multiplications in a single kernel launch:
```
C[b] = A[b] x B[b]  for b = 0, 1, ..., batch_size-1
```

```
Traditional (batch of 32):          Batched GEMM:
  Launch kernel 0  -> 7us overhead    Launch single kernel -> 7us overhead
  Launch kernel 1  -> 7us overhead    (processes all 32 matrices)
  ...                                 
  Launch kernel 31 -> 7us overhead    
  -----------------------------       -----------------------------
  Total: 32 x 7us = 224us overhead    Total: 7us overhead (32x less!)
```

### Benchmark Results (Tesla P100)

| Test Configuration        | Best Kernel           | GFLOPS  | Speedup |
|---------------------------|-----------------------|---------|---------|
| 64×64×64, batch=8         | Batched Basic         | 611     | 5.75×   |
| 256×256×256, batch=32     | Batched + RegBlock    | 5,030   | 4.01×   |
| 128×768×512, batch=32     | Batched + RegBlock    | 5,145   | 3.75×   |
| 1024×2048×3072, batch=32  | Batched + RegBlock    | **6,590** | 3.32×   |

### Key Insights

- **Small matrices**: Simple batching wins (5.75x) - launch overhead dominates
- **Large matrices**: Register blocking wins (3.32x) - compute dominates
- **Peak performance**: 6,590 GFLOPS at 71% of theoretical peak

### GPU Utilization

During large workload (`./matmul_batched 1024 2048 3072 32`):
- GPU Utilization: 100%
- Power Draw: 206W / 250W (82% TDP)
- Memory Used: 2,177 MiB / 16,384 MiB (13%)

### Use Cases

- Neural network batch processing
- Transformer attention heads (multi-head attention)
- LSTM/GRU gate computations
- Ensemble model inference

### Source Files

| File                                                                                   | Description          |
|----------------------------------------------------------------------------------------|----------------------|
| [`batched_gemm_phase3/src/matmul_batched.cu`](batched_gemm_phase3/src/matmul_batched.cu) | Batched GEMM kernels |
| [`batched_gemm_phase3/tests/test_batches.py`](batched_gemm_phase3/tests/test_batches.py) | Test suite           |

### Quick Start

```bash
cd cuda-matmul-phase3/src
nvcc -O3 -arch=sm_60 -o matmul_batched matmul_batched.cu

# Transformer attention heads (8 heads, 64x64 matrices)
./matmul_batched 64 64 64 8

# Neural network batch (32 samples, 256x256 layer)
./matmul_batched 256 256 256 32

# Large-scale workload (peak performance)
./matmul_batched 1024 2048 3072 32
```

---

## Phase 4: Tensor Cores (Future)

**Status**: o Future (requires Volta+ GPU)

Hardware-accelerated FP16xFP16->FP32 computation:
- Potential 8x speedup over FP32
- Available on V100, A100, H100
- P100 (Pascal) does not have Tensor Cores

---

## Phase 5: Multi-GPU (Future)

**Status**: o Future

Distributed matrix multiplication using NCCL:
- Scale beyond single GPU memory
- Linear scaling with GPU count
- Essential for very large matrices

---

## Documentation

### Technical Guides

| Document                                      | Description                     |
|-----------------------------------------------|---------------------------------|
| [**PROJECT_SUMMARY.md**](docs/PROJECT_SUMMARY.md)      | One-page project overview       |
| [**OPTIMIZATION_GUIDE.md**](docs/OPTIMIZATION_GUIDE.md) | Detailed optimization techniques|
| [**RESULTS.md**](docs/RESULTS.md)                      | Complete benchmark data         |

### Phase Documentation

| Document                              | Description                                |
|---------------------------------------|--------------------------------------------|
| [**Phase 2 README**](cuda-matmul-phase2/README.md)                     | Mixed precision overview and results       |
| [**Phase 2 Technical**](cuda-matmul-phase2/docs/PHASE2_MIXED_PRECISION.md) | Deep dive into FP16/FP32 precision         |
| [**Phase 3 README**](batched_gemm_phase3/README.md)                     | Batched GEMM overview and results          |
| [**Phase 3 Technical**](batched_gemm_phase3/docs/PHASE3_BATCHED_GEMM.md) | Deep dive into batched operations          |

---

## Hardware

- **GPU**: NVIDIA Tesla P100-PCIE-16GB
- **Architecture**: Pascal (sm_60)
- **CUDA Cores**: 3,584
- **Memory**: 16GB HBM2
- **Theoretical Peak**: 9,300 GFLOPS (FP32)
- **System**: RHEL9, CUDA 12.4

---

## Project Structure

```
cuda-matmul-optimization/
|
|-- README.md                          # This file
|-- matmul_kernels.cu                  # Base: Register blocking (6,436 GFLOPS)
|-- matmul_kernels_shared_memory.cu    # Base: Shared memory baseline
|
|-- docs/
|   |-- PROJECT_SUMMARY.md             # One-page overview
|   |-- OPTIMIZATION_GUIDE.md          # Technical deep dive
|   |-- RESULTS.md                     # Benchmark data
|   +-- PHASE2_MIXED_PRECISION.md      # Phase 2 copy (for reference)
|
|-- cuda-matmul-phase2/                # Phase 2: Mixed Precision
|   |-- README.md                      # Phase 2 overview
|   |-- src/
|   |   |-- matmul_mixed_precision.cu  # Mixed precision kernels
|   |   +-- test_mixed_precision.py    # Test suite
|   +-- docs/
|       +-- PHASE2_MIXED_PRECISION.md  # Technical documentation
|
+-- cuda-matmul-phase3/                # Phase 3: Batched GEMM
    |-- README.md                      # Phase 3 overview
    |-- src/
    |   +-- matmul_batched.cu          # Batched GEMM kernels
    |-- docs/
    |   +-- PHASE3_BATCHED_GEMM.md     # Technical documentation
    +-- tests/
        +-- test_batched.py            # Test suite
```

---

## Performance Summary

```
Performance Progression:

6,590 GFLOPS |-------------------+     Phase 3: Batched GEMM
             |                   |     71% of peak, NEW BEST!
6,436 GFLOPS |---------------+   |     Base: Register Blocking
             |               |   |     69% of peak, 92% of cuBLAS
4,731 GFLOPS |-----------+   |   |     Phase 2: Mixed Precision  
             |           |   |   |     2.6x speedup with FP16
1,817 GFLOPS |-------+   |   |   |     FP32 Baseline (2048^3)
             |       |   |   |   |
1,342 GFLOPS |-------+---+---+---+     Initial shared memory
             |
           0 +------------------------
              SM   FP32  Ph2  Base Ph3
```

---

## Quick Reference

### Compile Commands

```bash
# Base implementation
nvcc -O3 -arch=sm_60 -o matmul_register matmul_kernels.cu

# Phase 2: Mixed Precision
nvcc -O3 -arch=sm_60 -o matmul_mixed cuda-matmul-phase2/src/matmul_mixed_precision.cu

# Phase 3: Batched GEMM
nvcc -O3 -arch=sm_60 -o matmul_batched cuda-matmul-phase3/src/matmul_batched.cu
```

### Run Commands

```bash
# Base benchmarks
./matmul_register 1024 1024 1024
./matmul_register 16384 16384 16384

# Phase 2 benchmarks
./matmul_mixed 2048 2048 2048
./matmul_mixed 1024 1024 1024 100    # Overflow demo

# Phase 3 benchmarks
./matmul_batched 64 64 64 8          # Small matrices (5.75x speedup)
./matmul_batched 256 256 256 32      # Medium matrices
./matmul_batched 1024 2048 3072 32   # Large-scale (6,590 GFLOPS)
```

---

## Author

**Andrey Maltsev**  
GPU Computing & High-Performance Computing

**Contact**: andrey.maltsev@yahoo.com  
**Portfolio**: https://github.com/maltsev-andrey  
**LinkedIn**: https://www.linkedin.com/in/andrey-m-06189b91/

---

## License

MIT License - See individual files for details.

---

## Acknowledgments

- **NVIDIA** for CUDA toolkit and Tesla P100 hardware
- **GPU computing community** for optimization techniques
- **Open-source contributors** for inspiration

---

**Last Updated**: January 2025  
**Version**: 3.0 (Phase 3 Complete)
