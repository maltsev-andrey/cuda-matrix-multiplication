# Phase 3: Batched GEMM

![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia)
![Performance](https://img.shields.io/badge/Performance-6,590_GFLOPS-success)
![GPU](https://img.shields.io/badge/GPU-Tesla_P100-blue)

## Overview

Batched GEMM performs **multiple matrix multiplications in a single kernel launch**:

```
C[b] = A[b] × B[b]  for b = 0, 1, ..., batch_size-1
```

This eliminates kernel launch overhead and maximizes GPU utilization for workloads with many independent matrix operations.

## Why Batched GEMM Matters

### The Problem: Kernel Launch Overhead

```
Traditional approach (batch of 32 matrices):
  Launch kernel for C[0] = A[0] × B[0]  -> ~5-10 μs overhead
  Launch kernel for C[1] = A[1] × B[1]  -> ~5-10 μs overhead
  ...
  Launch kernel for C[31] = A[31] × B[31] -> ~5-10 μs overhead
  
  Total overhead: 32 x ~7 μs = ~224 μs
  For small matrices, overhead > computation time!

Batched approach:
  Launch single kernel for all 32 matrices -> ~7 μs overhead
  
  Overhead reduction: 32x less!
```

### Real-World Use Cases

| Application               | Batch Dimension               | Matrix Size            | Why Batched                     |
|---------------------------|-------------------------------|------------------------|---------------------------------|
| **Neural Network Batch**  | Batch size (32, 64, 128)      | Layer dimensions       | Process all samples together    |
| **Transformer Attention** | Number of heads (8, 12, 16)   | Sequence x Head dim    | Parallel head computation       |
| **Multi-Query Attention** | Number of queries             | Query x Key dimension  | Efficient attention             |
| **LSTM/GRU Gates**        | 4 gates × layers              | Hidden dimension       | Gate parallelism                |
| **Ensemble Models**       | Number of models              | Model dimensions       | Parallel inference              |

## Memory Layouts

### Layout 1: Strided (Contiguous Batches)

```
Memory: [A[0], A[1], ..., A[batch-1], B[0], B[1], ..., C[0], C[1], ...]

A_batch[b] = A + b * M * K
B_batch[b] = B + b * K * N
C_batch[b] = C + b * M * N

Pros: Simple indexing, good cache behavior
Cons: Requires contiguous allocation
```

### Layout 2: Array of Pointers

```
Memory: Matrices can be anywhere in memory
        A_ptrs = [ptr_A0, ptr_A1, ..., ptr_A31]
        B_ptrs = [ptr_B0, ptr_B1, ..., ptr_B31]
        C_ptrs = [ptr_C0, ptr_C1, ..., ptr_C31]

Pros: Flexible, non-contiguous batches
Cons: Extra pointer indirection
```

## Implementation Strategies

### Strategy 1: Batch in Grid Z-Dimension

```cuda
// Grid: (ceil(N/TILE), ceil(M/TILE), batch_size)
__global__ void batched_gemm(float* A, float* B, float* C, 
                              int M, int N, int K, int batch_size) {
    int batch = blockIdx.z;  // Batch index from z-dimension
    
    // Calculate per-batch pointers
    float* A_batch = A + batch * M * K;
    float* B_batch = B + batch * K * N;
    float* C_batch = C + batch * M * N;
    
    // Standard tiled matmul for this batch
    // ...
}
```

**Advantages:**
- Clean separation of batches
- No inter-batch synchronization needed
- Scales naturally with batch size

### Strategy 2: Combined with Register Blocking

```cuda
// Each thread computes 4×4 elements across batch dimension
__global__ void batched_gemm_register_blocked(...) {
    int batch = blockIdx.z;
    
    // Register accumulators for 4×4 output
    float sum[4][4] = {0.0f};
    
    // Load tiles, compute with register blocking
    // Same as Phase 1, but with batch offset
}
```

### Strategy 3: Mixed Precision Batched

```cuda
// FP16 storage + FP32 accumulation + batching
__global__ void batched_gemm_mixed(half* A, half* B, float* C, ...) {
    int batch = blockIdx.z;
    
    // FP16 shared memory tiles
    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];
    
    float sum = 0.0f;  // FP32 accumulator
    // ...
}
```

## Kernel Variants

| Kernel                                | Features                | Best For                     |
|---------------------------------------|-------------------------|------------------------------|
| `batched_gemm_basic`                  | Simple batching         | Baseline, debugging          |
| `batched_gemm_register_blocked`       | + Register blocking     | Large matrices               |
| `batched_gemm_mixed_precision`        | + FP16/FP32             | Memory-bound cases           |
| `batched_gemm_mixed_register_blocked` | All optimizations       | **Maximum performance**      |
| `batched_gemm_pointer_array`          | Non-contiguous memory   | Flexible layouts             |

## Expected Performance

### Small Matrices (128×128, batch=32)

```
Without batching: Many small kernels, high overhead
With batching:    Single kernel, full GPU utilization

Expected speedup: 2-5x from reduced launch overhead
```

### Larger Matrices (512×512, batch=8)

```
Without batching: Each matrix better utilizes GPU
With batching:    Still benefits from single launch

Expected speedup: 1.2-2x (less dramatic but still significant)
```

### Scaling Behavior

```
Batch Size    Speedup vs Loop    Reason
────────────────────────────────────────────
4             ~1.3x              Some overhead reduction
8             ~1.5x              Better GPU utilization  
16            ~2.0x              Good parallelism
32            ~2.5x              Excellent utilization
64            ~3.0x              Near-optimal batching
128+          ~3-4x              Diminishing returns
```

## Comparison with cuBLAS

cuBLAS provides batched GEMM functions:
- `cublasSgemmBatched` - Array of pointers
- `cublasSgemmStridedBatched` - Strided layout

Our implementation follows similar patterns and should achieve comparable performance for well-tuned cases.

## Usage

### Compile

```bash
nvcc -O3 -arch=sm_60 -o matmul_batched src/matmul_batched.cu
```

### Run

```bash
# Small matrices, large batch (attention-like)
./matmul_batched 64 64 64 16

# Medium matrices, medium batch
./matmul_batched 256 256 256 8

# Larger matrices, small batch
./matmul_batched 512 512 512 4

# Neural network layer simulation
./matmul_batched 1024 2048 3072 32
```

## Neural Network Context

### Transformer Self-Attention

```
Multi-Head Attention with 8 heads:

Q, K, V: [batch, seq_len, num_heads, head_dim]

For each head h:
  Attention[h] = softmax(Q[h] x K[h]^T / √d) x V[h]

Without batched GEMM: 8 x 2 = 16 kernel launches per layer
With batched GEMM:    2 kernel launches per layer (QxK^T and AttnxV)
```

### Batch Processing in MLP

```
Batch of 32 samples through linear layer:

Y = X x W  where X is [32, 768], W is [768, 3072]

Traditional: Process as single large matmul
Batched: Can split across multiple smaller operations if needed
```

## Integration with Previous Phases

### Building on Phase 1 (Rectangular)

Batched GEMM supports non-square matrices:
```bash
./matmul_batched 128 768 512 32   # M=128, K=768, N=512, batch=32
```

### Building on Phase 2 (Mixed Precision)

Combined kernel uses FP16 storage with FP32 accumulation:
```
batched_gemm_mixed_register_blocked:
  - FP16 input matrices (2x bandwidth)
  - FP32 accumulators (numerical stability)
  - Register blocking (16x work per thread)
  - Batching (single kernel launch)
```

## References

- [cuBLAS Batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference)
- [Optimizing Batched Matrix Operations](https://developer.nvidia.com/blog/cuda-pro-tip-optimizing-batch-computation/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
