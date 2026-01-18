# CUDA Matrix Multiplication Optimization

High-performance GPU-accelerated matrix multiplication achieving **6,436 GFLOPS** (69% of theoretical peak) on NVIDIA Tesla P100 through progressive CUDA optimization.

![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)
![Performance](https://img.shields.io/badge/Performance-6.4_TFLOPS-success)
![GPU](https://img.shields.io/badge/GPU-Tesla_P100-blue)

---

## Project Roadmap

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    CUDA Matrix Multiplication Phases                    	 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                         	 │
│ OK Base Implementation          Shared memory tiling, register blocking	 │
│    └── 6,436 GFLOPS (69% peak)  92% of cuBLAS efficiency               	 │
│                                                                        	 │
│ OK Phase 1: Rectangular         Support for M≠N≠K matrices            	 │
│    └── Integrated in base       Real-world matrix shapes              	 │
│                                                                       	 │
│ OK Phase 2: Mixed Precision     FP16 storage, FP32 accumulation         	 │
│    └── 4,731 GFLOPS             2.6× speedup, 2× bandwidth          		 │
│                                                                         	 │
│  ○ Phase 3: Batched GEMM        Multiple matrices per kernel           	 │
│    └── Planned                  Essential for neural networks        		 │
│                                                                     	     │
│  ○ Phase 4: Tensor Cores        Volta+ GPU acceleration              		 │
│    └── Future                   Potential 8× speedup                 		 │
│                                                                 	         │
│  ○ Phase 5: Multi-GPU           NCCL distributed computing          	     │
│    └── Future                   Scale beyond single GPU             	     │
│                                                                     	     │
│  Legend: OK Complete  ○ Planned                                      	     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Overview

| Phase                                                    | Status    | Performance   | Description                       |
|----------------------------------------------------------|-----------|---------------|-----------------------------------|
| [**Base Implementation**](#base-implementation)          | Complete  | 6,436 GFLOPS  | Shared memory + register blocking |
| [**Phase 1: Rectangular**](#phase-1-rectangular-matrices)| Complete  | Integrated    | Support for M≠N≠K                 |
| [**Phase 2: Mixed Precision**](cuda-matmul-phase2/README.md) | Complete  | 4,731 GFLOPS  | FP16/FP32 mixed precision         |
| [Phase 3: Batched GEMM](#phase-3-batched-gemm-planned)   | ○ Planned | –             | Multiple matrix pairs             |
| [Phase 4: Tensor Cores](#phase-4-tensor-cores-future)    | ○ Future  | –             | Hardware FP16 acceleration        |
| [Phase 5: Multi-GPU](#phase-5-multi-gpu-future)          | ○ Future  | –             | Distributed computing             |

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
| 1     | Shared Memory Tiling (16×16)  | 1,732 GFLOPS | Baseline    |
| 2     | Larger Tiles (32×32)          | 2,080 GFLOPS | +55%        |
| 3     | Register Blocking (4×4)       | 6,436 GFLOPS | **+379%**   |

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

Support for non-square matrices: `C(M×N) = A(M×K) × B(K×N)`

Already supported in `matmul_kernels.cu` with proper boundary handling for M≠N≠K.

```bash
./matmul_register 2048 512 4096    # Rectangular matrices
```

---

## Phase 2: Mixed Precision

**Status**: Complete | **Performance**: 4,731 GFLOPS | **Speedup**: 2.6× vs FP32

**[Full Documentation](cuda-matmul-phase2/README.md)**

### Key Results

```
┌─────────────────────────────────────────────────────────────┐
│  FP16 storage  → 2× memory bandwidth                        │
│  FP32 accumulate → Maintains numerical accuracy             │
│  = Best of both worlds!                                     │
└─────────────────────────────────────────────────────────────┘
```

| Matrix   | Method                        | GFLOPS  | vs FP32 |
|----------|-------------------------------|---------|---------|
| 2048³    | FP32 Baseline                 | 1,817   | 1.00×   |
| 2048³    | **Mixed + Register Blocked**  | **4,731** | **2.60×** |

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

## Phase 3: Batched GEMM (Planned)

**Status**: ○ Planned

Multiple matrix multiplications in a single kernel launch:
```
C[b] = A[b] × B[b]  for b = 0, 1, ..., batch_size-1
```

**Use Cases**:
- Neural network batch processing
- Transformer attention heads
- Parallel inference

---

## Phase 4: Tensor Cores (Future)

**Status**: ○ Future (requires Volta+ GPU)

Hardware-accelerated FP16×FP16→FP32 computation:
- Potential 8× speedup over FP32
- Available on V100, A100, H100
- P100 (Pascal) does not have Tensor Cores

---

## Phase 5: Multi-GPU (Future)

**Status**: ○ Future

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
│
├── README.md                          # This file
├── matmul_kernels.cu                  # Base: Register blocking (6,436 GFLOPS)
├── matmul_kernels_shared_memory.cu    # Base: Shared memory baseline
│
├── docs/
│   ├── PROJECT_SUMMARY.md             # One-page overview
│   ├── OPTIMIZATION_GUIDE.md          # Technical deep dive
│   ├── RESULTS.md                     # Benchmark data
│   └── PHASE2_MIXED_PRECISION.md      # Phase 2 copy (for reference)
│
└── cuda-matmul-phase2/                # Phase 2: Mixed Precision
    ├── README.md                      # Phase 2 overview
    ├── src/
    │   ├── matmul_mixed_precision.cu  # Mixed precision kernels
    │   └── test_mixed_precision.py    # Test suite
    └── docs/
        └── PHASE2_MIXED_PRECISION.md  # Technical documentation
```

---

## Performance Summary

```
Performance Progression:

6,436 GFLOPS ├───────────────────┐  Base: Register Blocking
             │                   │  69% of peak, 92% of cuBLAS
4,731 GFLOPS ├─────────────┐     │  Phase 2: Mixed Precision  
             │             │     │  2.6× speedup with FP16
1,817 GFLOPS ├───────┐     │     │  FP32 Baseline (2048³)
             │       │     │     │
1,342 GFLOPS ├───────┴─────┴─────┘  Initial shared memory
             │
           0 └────────────────────
             Base    Ph2    Peak
```

---

## Quick Reference

### Compile Commands

```bash
# Base implementation
nvcc -O3 -arch=sm_60 -o matmul_register matmul_kernels.cu

# Phase 2: Mixed Precision
nvcc -O3 -arch=sm_60 -o matmul_mixed cuda-matmul-phase2/src/matmul_mixed_precision.cu
```

### Run Commands

```bash
# Base benchmarks
./matmul_register 1024 1024 1024
./matmul_register 16384 16384 16384

# Phase 2 benchmarks
./matmul_mixed 2048 2048 2048
./matmul_mixed 1024 1024 1024 100    # Overflow demo
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
**Version**: 2.0 (Phase 2 Complete)
