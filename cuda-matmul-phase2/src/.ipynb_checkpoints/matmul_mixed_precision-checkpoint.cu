/**
 * CUDA Matrix Multiplication - Phase 2: Mixed Precision
 * 
 * Implements mixed-precision matrix multiplication:
 * - FP16 (half) for storage and memory transfer
 * - FP32 (float) for accumulation to maintain numerical stability
 * 
 * This approach provides:
 * - 2× memory bandwidth improvement (FP16 is half the size)
 * - Maintained numerical accuracy (FP32 accumulation)
 * - Foundation for Tensor Core utilization (future enhancement)
 * 
 * Connection to Floating-Point Forensics:
 * - Demonstrates why accumulation precision matters
 * - Shows catastrophic cancellation in pure FP16
 * - Illustrates the precision/performance tradeoff
 * 
 * Author: Andrey Maltsev
 * Phase 2 of Matrix Multiplication Enhancement Project
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h> // for half precision support
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// Configuration constants
#define TILE_SIZE 16
#define THREAD_TILE 4

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit (EXIT_FAILURE); \
        } \
    } while (0)


// ============================================================================
// EDUCATIONAL: IEEE 754 HALF PRECISION (FP16) LAYOUT
// ============================================================================
/*
 * FP16 (half precision) bit layout:
 * 
 *   Sign  Exponent   Mantissa
 *    1      5          10      = 16 bits total
 *   [S][EEEEE][MMMMMMMMMM]
 * 
 * Compared to FP32:
 *    1      8          23      = 32 bits total
 *   [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
 * 
 * FP16 Characteristics:
 * - Range: ~6.0 × 10^-8 to 65504
 * - Precision: ~3 decimal digits
 * - Smallest positive normal: 2^-14 ≈ 6.1 × 10^-5
 * 
 * FP32 Characteristics:
 * - Range: ~1.2 × 10^-38 to 3.4 × 10^38
 * - Precision: ~7 decimal digits
 * 
 * WHY MIXED PRECISION WORKS:
 * - Memory bandwidth is often the bottleneck
 * - FP16 halves memory traffic (2 bytes vs 4 bytes)
 * - Accumulation in FP32 prevents error accumulation
 * - Final result maintains FP32 accuracy
 */


// ============================================================================
// KERNEL 1: PURE FP32 (Baseline for comparison)
// ============================================================================
__global__ void matmul_fp32 (
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K
)    {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int tx = threadIdx.x, ty = threadIdx.y;
        int row = blockIdx.y * TILE_SIZE + ty;
        int col = blockIdx.x * TILE_SIZE + tx;

        float sum = 0.0f;
        int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

        for (int t = 0; t < numTiles; t++) {
            int a_col = t * TILE_SIZE + tx;
            int b_row = t * TILE_SIZE + ty;

            As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
            Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[ty][k] * Bs[k][tx];
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
}


// ============================================================================
// KERNEL 2: PURE FP16 (Demonstrates precision problems)
// ============================================================================
/*
 * This kernel intentionally uses FP16 for accumulation to demonstrate
 * the numerical instability that occurs. Use this for EDUCATIONAL purposes
 * to understand why mixed precision is necessary.
 * 
 * Problems with pure FP16 accumulation:
 * 1. Overflow: Values > 65504 become inf
 * 2. Underflow: Small gradients vanish
 * 3. Rounding: Only ~3 decimal digits of precision
 * 4. Catastrophic cancellation: When adding small to large values
 */
__global__ void matmul_fp16_pure(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // DANGER: FP16 accumulator - will lose precision!
    half sum = __float2half(0.0f);
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : __float2half(0.0f);
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : __float2half(0.0f);

        __syncthreads();

         # pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // FP16 multiply and accumulate - precision loss here!
            sum = __hadd(sum, __hmul(As[ty][k], Bs[k][tx]));
        }

        __syncthreads();
    }

    if (row < M && col < N) {
            C[row * N + col] = sum;
    }
}


// ============================================================================
// KERNEL 3: MIXED PRECISION (FP16 storage, FP32 accumulation)
// ============================================================================
/*
 * THE RECOMMENDED APPROACH:
 * - Load FP16 values from global memory (2× bandwidth)
 * - Convert to FP32 for computation
 * - Accumulate in FP32 (maintains precision)
 * - Store result as FP32 (or convert back to FP16 if needed)
 * 
 * This is the strategy used by:
 * - NVIDIA Tensor Cores (hardware FP16×FP16→FP32)
 * - PyTorch AMP (Automatic Mixed Precision)
 * - TensorFlow mixed precision training
 */
__global__ void matmul_mixed_precision(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
)  {
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // FP32 accumulator 9 this is the key to numerical stability
    float sum  = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        //  Load as FP16 (memory efficient)
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : __float2half(0.0f);
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : __float2half(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Convert to FP32 for computation, maintains precision
            float a_val = __half2float(As[ty][k]);
            float b_val = __half2float(Bs[k][tx]);
            sum += a_val * b_val; // FP32 accumulation
        }

        __syncthreads();
    }

    // Output as FP32 (full precision results)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// ============================================================================
// KERNEL 4: MIXED PRECISION WITH REGISTER BLOCKING (Optimized)
// ============================================================================
__global__ void matmul_mixed_precision_register_blocked(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half As[TILE_SIZE * THREAD_TILE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE * THREAD_TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int baseRow = blockIdx.y * (TILE_SIZE * THREAD_TILE) + ty;
    int baseCol = blockIdx.x * (TILE_SIZE * THREAD_TILE) + tx;

    // FP32 accumulators in registers
    float sum[THREAD_TILE][THREAD_TILE];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            sum[i][j] = 0.0f;
        }
    }
    
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load A tile (FP16)
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            int row = baseRow + i * TILE_SIZE;
            int col = t * TILE_SIZE + tx;
            As[ty + i * TILE_SIZE][tx] = (row < M && col < K) ?
                A[row * K + col] : __float2half(0.0f);
        }

        // Load B tile (FP16)    
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = t * TILE_SIZE + ty;
            int col = baseCol + j * TILE_SIZE;
            Bs[ty][tx + j     * TILE_SIZE] = (row < K && col < N) ?
                B[row * N + col] : __float2half(0.0f);
        }

        __syncthreads();

        // Compute with FP32 accumulation
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load  A value and convert to FP32
            float a_reg[THREAD_TILE];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                a_reg[i] = __half2float(As[ty + i * TILE_SIZE][k]) ;
            }

            // Compute with FP32
            #pragma unroll
            for  (int j = 0; j < THREAD_TILE; j++) {
                float b_val = __half2float(Bs[k][tx + j * TILE_SIZE]);
                #pragma unroll
                for (int i = 0; i < THREAD_TILE; i++) {
                    sum[i][j] += a_reg[i] * b_val;
                }
            }
        }

        __syncthreads();
    }

    // Write FP32 results
    #pragma unroll
    for (int i =0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = baseRow + i * TILE_SIZE;
            int col = baseCol + j * TILE_SIZE;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}


// ============================================================================
// KERNEL 5: MIXED PRECISION WITH OUTPUT AS FP16
// ============================================================================
/*
 * For cases where you need FP16 output (e.g., feeding into next layer)
 * Still uses FP32 accumulation internally for accuracy
 */
__global__ void matmul_mixed_precision_fp16_output(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f; // FP32 accumulator
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : __float2half(0.0f);
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : __float2half(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        // Convert back to FP16 for output
        C[row * N + col] = __float2half(sum);
    }
}


// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Convert FP32 array to FP16
__global__ void convert_fp32_to_fp16(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// Convert FP16 array to FP32
__global__ void convert_fp16_to_fp32(const half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

// Initialize matrix with random values (CPU)
void init_matrix_fp32(float* mat, int size, float scale = 1.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

// CPU reference (FP32)
void matmul_cpu_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Compute error statistics
void compute_error_stats(const float* result, const float* reference, int size,
                                        float* max_abs_error, float* max_rel_error,
                                        float* mean_abs_error, float* rms_error) {
    double sum_abs = 0.0, sum_sq = 0.0;
    *max_abs_error = 0.0f;
    *max_rel_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float abs_err = fabsf(result[i] - reference[i]);
        float rel_err = abs_err / (fabsf(reference[i]) + 1e-8f);

        if (abs_err > *max_abs_error) *max_abs_error = abs_err;
        if (rel_err > *max_rel_error) *max_rel_error = rel_err;

        sum_abs += abs_err;
        sum_sq += abs_err * abs_err;
    }

    *mean_abs_error = (float)(sum_abs / size);
    *rms_error = (float)sqrt(sum_sq / size);
}


// ============================================================================
// BENCHMARK FUNCTION
// ============================================================================
typedef void (*KernelFuncFP32)(const float*, const float*, float*, int, int, int);
typedef void (*KernelFuncMixed)(const half*, const half*, float*, int, int, int);
typedef void (*KernelFuncFP16)(const half*, const half*, half*, int, int, int);

template<typename KernelFunc, typename InputType, typename OutputType>
float benchmark_kernel(
    KernelFunc kernel,
    const char* name,
    InputType* d_A, InputType* d_B, OutputType* d_C,
    int M, int N, int K,
    dim3 grid, dim3 block,
    int warmup = 3, int runs = 10
) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / runs;
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    
    printf("%-35s: %8.3f ms, %8.2f GFLOPS\n", name, avg_ms, gflops);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return gflops;
}


// ============================================================================
// MAIN PROGRAM
// ============================================================================
void print_usage(const char* prog) {
    printf("Usage: %s M K N [scale]\n", prog);
    printf(" M, K, N: Matrix dimensions C(M*N) = A(M*K) * B(K*N)\n");
    printf("  scale: Value range scale (default 1.0, use larger to trigger FP16 issues)\n");
    printf("\nExamples:\n");
    printf("  %s 1024 1024 1024          # Normal range\n", prog);
    printf("  %s 1024 1024 1024 100      # Large values (shows FP16 overflow)\n", prog);
    printf("  %s 2048 2048 2048 0.01     # Small values (shows FP16 underflow)\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    float scale = (argc > 4) ? atof(argv[4]) : 1.0f;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║        CUDA Matrix Multiplication - Mixed Precision              ║\n");
    printf("║                         Phase 2                                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Configuration:\n");
    printf("  Matrix: C(%d×%d) = A(%d×%d) × B(%d×%d)\n", M, N, M, K, K, N);
    printf("  Value scale: %.2f (range: [%.2f, %.2f])\n", scale, -scale, scale);
    printf("  FLOPs: %.2e\n\n", 2.0 * M * N * (double)K);

    // Memory sizes
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    printf("Memory usage:\n");
    printf(" FP32: A=%.1fMB, B=%.1fMB, C=%.1fMB (Total: %.1fMB)\n",
          size_A * 4 / 1e6, size_B * 4 / 1e6, size_C * 4 / 1e6,
          (size_A + size_B + size_C) * 4 / 1e6);
    printf(" FP16: A=%.1fMB, B=%.1fMB (Total: %.1fMB) - 2* bandwidth!\n\n",
          size_A * 2 / 1e6, size_B * 2 / 1e6, (size_A + size_B) * 2 / 1e6);

    // Allocate host memory
    float* h_A = (float*)malloc(size_A * sizeof(float));
    float* h_B = (float*)malloc(size_B * sizeof(float));
    float* h_C_ref = (float*)malloc(size_C * sizeof(float));
    float* h_C_result = (float*)malloc(size_C * sizeof(float));
    
    // Initialize
    srand(42);
    init_matrix_fp32(h_A, size_A, scale);
    init_matrix_fp32(h_B, size_B, scale);
    
    // Allocate device memory
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    float *d_C_mixed;
    
    CUDA_CHECK(cudaMalloc(&d_A_fp32, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_fp32, size_C * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_A_fp16, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, size_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_fp16, size_C * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_mixed, size_C * sizeof(float)));
    
    // Copy and convert data
    CUDA_CHECK(cudaMemcpy(d_A_fp32, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp32, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));
    
    // Convert to FP16
    dim3 conv_block(256);
    dim3 conv_grid_A((size_A + 255) / 256);
    dim3 conv_grid_B((size_B + 255) / 256);
    convert_fp32_to_fp16<<<conv_grid_A, conv_block>>>(d_A_fp32, d_A_fp16, size_A);
    convert_fp32_to_fp16<<<conv_grid_B, conv_block>>>(d_B_fp32, d_B_fp16, size_B);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Grid configurations
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    int block_output = TILE_SIZE * THREAD_TILE;
    dim3 grid_reg((N + block_output - 1) / block_output, (M + block_output - 1) / block_output);

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("                       PERFORMANCE BENCHMARK                       \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Benchmark all kernels
    float gflops_fp32 = benchmark_kernel(
        matmul_fp32, "FP32 (baseline)",
        d_A_fp32, d_B_fp32, d_C_fp32, M, N, K, grid, block);

    float gflops_fp16_pure = benchmark_kernel(
        matmul_fp16_pure, "FP16 PURE (unstable)",
        d_A_fp16, d_B_fp16, d_C_fp16, M, N, K, grid, block);

    float gflops_mixed = benchmark_kernel(
        matmul_mixed_precision, "Mixed (FP16 in, FP32 accum)",
        d_A_fp16, d_B_fp16, d_C_mixed, M, N, K, grid, block);
    
    float gflops_mixed_reg = benchmark_kernel(
        matmul_mixed_precision_register_blocked, "Mixed + Register Blocked",
        d_A_fp16, d_B_fp16, d_C_mixed, M, N, K, grid_reg, block);
    
    float gflops_mixed_fp16out = benchmark_kernel(
        matmul_mixed_precision_fp16_output, "Mixed (FP16 in, FP16 out)",
        d_A_fp16, d_B_fp16, d_C_fp16, M, N, K, grid, block);
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("                       NUMERICAL ACCURACY                                   \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    // Compute CPU reference
    printf("Computing CPU reference...\n");
    matmul_cpu_fp32(h_A, h_B, h_C_ref, M, N, K);

    float max_abs, max_rel, mean_abs, rms;

    // Check FP32 accuracy
    CUDA_CHECK(cudaMemcpy(h_C_result, d_C_fp32, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    compute_error_stats(h_C_result, h_C_ref, size_C, &max_abs, &max_rel, &mean_abs, &rms);
    printf("\nFP32 Baseline:\n");
    printf("  Max Abs Error: %.6e\n", max_abs);
    printf("  Max Rel Error: %.6e\n", max_rel);
    printf("  RMS Error:     %.6e\n", rms);
    
    // Check FP16 Pure accuracy (convert FP16 output to FP32 for comparison)
    // float* h_C_fp16_as_fp32 = (float*)malloc(size_C * sizeof(float));
    dim3 conv_grid_C((size_C + 255) / 256);
    
    // Run FP16 pure one more time to get result
    matmul_fp16_pure<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_fp16, M, N, K);
    convert_fp16_to_fp32<<<conv_grid_C, conv_block>>>(d_C_fp16, d_C_mixed, size_C);
    CUDA_CHECK(cudaMemcpy(h_C_result, d_C_mixed, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    compute_error_stats(h_C_result, h_C_ref, size_C, &max_abs, &max_rel, &mean_abs, &rms);
    printf("\nFP16 Pure (UNSTABLE - for demonstration):\n");
    printf("  Max Abs Error: %.6e  ← MUCH LARGER!\n", max_abs);
    printf("  Max Rel Error: %.6e\n", max_rel);
    printf("  RMS Error:     %.6e\n", rms);

    // Check overflow/underflow
    int inf_count = 0, nan_count = 0;
    for (int i = 0; i < size_C; i++) {
        if (isinf(h_C_result[i])) inf_count++;
        if (isnan(h_C_result[i])) nan_count++;
    }
    if (inf_count > 0 || nan_count > 0) {
        printf("  OVERFLOW DETECTED: %d inf, %d nan values!\n", inf_count, nan_count);
    }
    
    // Check Mixed Precision accuracy
    matmul_mixed_precision<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_mixed, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_C_result, d_C_mixed, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    compute_error_stats(h_C_result, h_C_ref, size_C, &max_abs, &max_rel, &mean_abs, &rms);
    printf("\nMixed Precision (FP16 storage, FP32 accumulation):\n");
    printf("  Max Abs Error: %.6e  ← Close to FP32!\n", max_abs);
    printf("  Max Rel Error: %.6e\n", max_rel);
    printf("  RMS Error:     %.6e\n", rms);
    
    // Check Mixed + Register Blocked accuracy
    matmul_mixed_precision_register_blocked<<<grid_reg, block>>>(d_A_fp16, d_B_fp16, d_C_mixed, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_C_result, d_C_mixed, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    compute_error_stats(h_C_result, h_C_ref, size_C, &max_abs, &max_rel, &mean_abs, &rms);
    printf("\nMixed + Register Blocked:\n");
    printf("  Max Abs Error: %.6e\n", max_abs);
    printf("  Max Rel Error: %.6e\n", max_rel);
    printf("  RMS Error:     %.6e\n", rms);
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("                           SUMMARY                                \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("Performance:\n");
    printf("  Mixed vs FP32:     %.2fx speedup (memory bandwidth)\n", gflops_mixed / gflops_fp32);
    printf("  Mixed+Reg vs FP32: %.2fx speedup\n", gflops_mixed_reg / gflops_fp32);
    
    printf("\nKey Insight:\n");
    printf("  FP16 storage reduces memory traffic by 2×\n");
    printf("  FP32 accumulation maintains numerical accuracy\n");
    printf("  Best of both worlds: speed AND precision!\n");
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("              CONNECTION TO FLOATING-POINT FORENSICS                \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("What we observed:\n");
    printf("1. FP16 Pure shows precision loss due to:\n");
    printf("   - Limited mantissa (10 bits vs 23 bits)\n");
    printf("   - Accumulation of rounding errors\n");
    printf("   - Catastrophic cancellation when adding small to large\n");
    printf("\n");
    printf("2. Mixed Precision maintains accuracy because:\n");
    printf("   - FP32 accumulator has 23-bit mantissa\n");
    printf("   - Intermediate results don't lose precision\n");
    printf("   - Only final conversion has rounding error\n");
    printf("\n");
    printf("3. Try with scale=100 to see FP16 overflow (values > 65504)\n");
    printf("   Try with scale=0.001 to see FP16 underflow/denormal issues\n");
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_result);
    //free(h_C_fp16_as_fp32);
    
    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_C_fp32));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_C_fp16));
    CUDA_CHECK(cudaFree(d_C_mixed));
    
    printf("\n   Complete!\n\n");
    
    return 0;
}
