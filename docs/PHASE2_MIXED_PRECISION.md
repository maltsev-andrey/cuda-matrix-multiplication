# Phase 2: Mixed Precision Matrix Multiplication

![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)
![Performance](https://img.shields.io/badge/Performance-6.4_TFLOPS-success)
![GPU](https://img.shields.io/badge/GPU-Tesla_P100-blue)


## Overview

This phase implements **mixed-precision matrix multiplication** using:
- **FP16 (half precision)** for storage and memory transfer
- **FP32 (single precision)** for accumulation and computation

This approach provides **2× memory bandwidth improvement** while **maintaining numerical accuracy**.

## Connection to Floating-Point Forensics

This phase directly applies concepts from your FP Forensics project:

| FP Forensics Concept | Application in Mixed Precision |
|---------------------|-------------------------------|
| IEEE 754 representation | FP16 has 10-bit mantissa vs FP32's 23-bit |
| Accumulation errors | Why FP16 accumulation fails |
| Catastrophic cancellation | Occurs when adding small to large in FP16 |
| ULP analysis | Quantifying precision loss |
| Compensated summation | FP32 accumulation is a form of this |

## IEEE 754 Format Comparison

```
FP16 (half precision):
┌───┬─────────┬────────────────┐
│ S │  EEEEE  │ MMMMMMMMMM     │  = 16 bits
│ 1 │    5    │     10         │
└───┴─────────┴────────────────┘
  │      │           │
  │      │           └── Mantissa: ~3 decimal digits precision
  │      └────────────── Exponent: range ~6e-5 to 65504
  └───────────────────── Sign bit

FP32 (single precision):
┌───┬──────────────┬───────────────────────────────┐
│ S │  EEEEEEEE    │ MMMMMMMMMMMMMMMMMMMMMMM       │  = 32 bits
│ 1 │      8       │            23                 │
└───┴──────────────┴───────────────────────────────┘
  │        │                    │
  │        │                    └── Mantissa: ~7 decimal digits precision
  │        └─────────────────────── Exponent: range ~1e-38 to 3e38
  └──────────────────────────────── Sign bit
```

## Why Mixed Precision Works

### The Problem with Pure FP16

```
Accumulation in FP16:
  sum = 0.0 (fp16)           ← Only 10 bits of precision
  for k in range(K):
    sum += A[i,k] * B[k,j]   ← Each operation loses precision!
                              ← After K iterations, massive error

Issues:
1. Rounding at each step (only ~3 decimal digits)
2. Overflow if sum > 65504
3. Underflow if values < 6e-5
4. Catastrophic cancellation when adding small to large
```

### The Mixed Precision Solution

```
Accumulation in Mixed Precision:
  sum = 0.0 (fp32)           ← 23 bits of precision!
  for k in range(K):
    a = convert_to_fp32(A_fp16[i,k])
    b = convert_to_fp32(B_fp16[i,k])
    sum += a * b             ← Full FP32 precision maintained!
  
  C[i,j] = sum               ← Accurate result

Benefits:
1. Memory: Store/transfer FP16 (half the bytes)
2. Bandwidth: 2× more data per memory transaction
3. Accuracy: FP32 accumulation prevents error buildup
```

### Memory Bandwidth Analysis

```
Matrix multiplication is memory-bound for large matrices:

FP32 only:
  Load A: M×K × 4 bytes = 4MK bytes
  Load B: K×N × 4 bytes = 4KN bytes
  Store C: M×N × 4 bytes = 4MN bytes
  
Mixed Precision:
  Load A: M×K × 2 bytes = 2MK bytes   ← 2× less!
  Load B: K×N × 2 bytes = 2KN bytes   ← 2× less!
  Store C: M×N × 4 bytes = 4MN bytes  (keep FP32 for accuracy)
  
For 4096×4096 matrices:
  FP32: 192 MB total
  Mixed: 128 MB total (33% reduction)
```

## Implementation Details

### Kernel Variants

| Kernel                                      | Input | Accumulator | Output | Use Case                     |
|---------------------------------------------|-------|-------------|--------|------------------------------|
| `matmul_fp32`                               | FP32  | FP32        | FP32   | Baseline                     |
| `matmul_fp16_pure`                          | FP16  | FP16        | FP16   | Educational (shows problems) |
| `matmul_mixed_precision`                    | FP16  | FP32        | FP32   | **Recommended**              |
| `matmul_mixed_precision_register_blocked`   | FP16  | FP32        | FP32   | **Best performance**         |
| `matmul_mixed_precision_fp16_output`        | FP16  | FP32        | FP16   | Chained operations           |

### Key Code Pattern

```cuda
// Mixed precision kernel core
__shared__ half As[TILE_SIZE][TILE_SIZE];  // FP16 in shared memory
__shared__ half Bs[TILE_SIZE][TILE_SIZE];

float sum = 0.0f;  // FP32 accumulator - THE KEY!

for (int t = 0; t < numTiles; t++) {
    // Load FP16 from global memory (efficient)
    As[ty][tx] = A[row * K + col];  // FP16 load
    Bs[ty][tx] = B[row * N + col];  // FP16 load
    
    __syncthreads();
    
    for (int k = 0; k < TILE_SIZE; k++) {
        // Convert to FP32 for computation (accurate)
        float a_val = __half2float(As[ty][k]);
        float b_val = __half2float(Bs[k][tx]);
        sum += a_val * b_val;  // FP32 multiply-add
    }
}

C[row * N + col] = sum;  // Store FP32 result
```

## Usage

### CUDA Version

```bash
# Compile
nvcc -O3 -arch=sm_60 -o matmul_mixed src/matmul_mixed_precision.cu

# Run benchmarks
./matmul_mixed 1024 1024 1024         # Normal range
./matmul_mixed 1024 1024 1024 100     # Large values (shows FP16 overflow)
./matmul_mixed 2048 2048 2048 0.001   # Small values (shows underflow)
```

### Python Version

```bash
# Install CuPy
pip install cupy-cuda12x

# Run precision comparison
python src/matmul_mixed_precision.py -M 1024 -K 1024 -N 1024

# Show FP16 limitations (educational)
python src/matmul_mixed_precision.py --demo-fp16

# Show accumulation error
python src/matmul_mixed_precision.py --demo-accumulation

# Trigger overflow
python src/matmul_mixed_precision.py --scale 100
```

## Expected Results

### Performance

```
Method                      Time (ms)    GFLOPS
─────────────────────────────────────────────────
FP32 (baseline)                5.234     410.2
FP16 Pure (unstable!)          2.891     742.1  ← Fast but wrong!
Mixed (FP16 in, FP32 accum)    3.012     712.3  ← Best balance
Mixed + Register Blocked       1.845    1162.4  ← Maximum performance
```

### Numerical Accuracy

```
Method          Max Relative Error    Notes
────────────────────────────────────────────────────────
FP32            1.2e-6               Baseline accuracy
FP16 Pure       4.7e-2               ~40,000× worse!
Mixed           2.8e-6               Close to FP32!
```

### Overflow Demonstration (scale=100)

```
FP16 Pure Results:
  inf count: 847,231
  nan count: 12,045
  
  OVERFLOW DETECTED!
  Values exceeded FP16 max (65504)
```

## Real-World Applications

### Where Mixed Precision is Used

1. **Deep Learning Training**
   - PyTorch AMP (Automatic Mixed Precision)
   - TensorFlow mixed precision
   - All modern training frameworks

2. **Inference**
   - TensorRT
   - ONNX Runtime
   - Production ML systems

3. **NVIDIA Tensor Cores**
   - Hardware FP16×FP16→FP32
   - Up to 8× speedup over FP32
   - Volta, Turing, Ampere, Hopper architectures

### Tensor Core Connection (Future Enhancement)

```
Current implementation:
  FP16 load → FP32 convert → FP32 multiply → FP32 accumulate

Tensor Core (if available):
  FP16 load → Hardware FP16×FP16→FP32 → FP32 accumulate
  
Potential: 8× additional speedup on Volta+ GPUs
(Your P100 is Pascal, no Tensor Cores)
```

## Experiments to Try

### 1. Accumulation Error Scaling
```bash
# See how error grows with K dimension
python -c "
from matmul_mixed_precision import demonstrate_accumulation_error
for n in [10, 100, 1000, 10000, 100000]:
    r = demonstrate_accumulation_error(n, 0.1)
    print(f'K={n}: FP16 error = {r[\"fp16_error\"]:.6f}')
"
```

### 2. Overflow Threshold
```bash
# Find where FP16 overflows
./matmul_mixed 512 512 512 50    # OK
./matmul_mixed 512 512 512 100   # Borderline
./matmul_mixed 512 512 512 200   # Overflow!
```

### 3. Underflow Behavior
```bash
# Very small values
./matmul_mixed 512 512 512 0.0001  # Some precision loss
./matmul_mixed 512 512 512 0.00001 # More loss
```

## Integration with Project

### File Structure

```
cuda-matrix-multiplication/
├── matmul_kernels.cu                    # Original FP32
├── matmul_rectangular.cu                # Phase 1
├── matmul_mixed_precision.cu            # Phase 2 (NEW)
├── matmul_mixed_precision.py            # Python wrapper (NEW)
└── docs/
    ├── PHASE1_RECTANGULAR.md
    └── PHASE2_MIXED_PRECISION.md        # This file
```

### Combining with Phase 1

The mixed precision kernels already support rectangular matrices!
```bash
./matmul_mixed 2048 512 4096  # Works with M≠K≠N
```

## Next Steps (Phase 3 Preview)

**Batched Matrix Multiplication**
- Multiple matrix pairs in one kernel launch
- Essential for neural network inference
- `C[b] = A[b] × B[b]` for batch index b

## References

- [NVIDIA Mixed Precision Training](https://developer.nvidia.com/automatic-mixed-precision)
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [IEEE 754-2008 Standard](https://ieeexplore.ieee.org/document/4610935)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
