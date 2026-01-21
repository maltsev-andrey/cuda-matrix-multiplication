/**
 * CUDA Matrix Multiplication - Phase 3: Batched GEMM
 * 
 * Performs multiple matrix multiplications in a single kernel launch:
 *   C[b] = A[b] × B[b]  for b = 0, 1, ..., batch_size-1
 * 
 * Use Cases:
 * - Neural network batch processing (forward/backward pass)
 * - Transformer attention heads (parallel head computation)
 * - Multi-instance inference
 * - Parallel solving of independent linear systems
 * 
 * Strategies Implemented:
 * 1. Naive batched: One kernel launch per matrix (baseline)
 * 2. Batched kernel: Single kernel, batch in grid z-dimension
 * 3. Strided batched: Contiguous memory layout with strides
 * 4. Array of pointers: Flexible non-contiguous batches
 * 
 * Author: Andrey Maltsev
 * Phase 3 of Matrix Multiplication Enhancement Project
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Configuration
#define TILE_SIZE 16
#define THREAD_TILE 4

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// ============================================================================
// EDUCATIONAL: BATCHED GEMM
// ============================================================================
/*
 * In neural networks, we often need to compute many small matrix multiplications:
 * 
 * Example 1: Batch Processing
 * ---------------------------
 * Input: batch of 32 images, each 224×224×3
 * First layer: 32 × (input × weights) operations
 * Without batching: 32 kernel launches (overhead!)
 * With batching: 1 kernel launch (efficient!)
 * 
 * Example 2: Transformer Attention
 * --------------------------------
 * Multi-head attention with 8 heads:
 *   Q[h] × K[h]^T for h = 0..7 (8 independent matmuls)
 * Without batching: 8 kernel launches
 * With batching: 1 kernel launch
 * 
 * Example 3: Recurrent Networks
 * -----------------------------
 * LSTM/GRU gates computed in parallel
 * Multiple weight matrices applied simultaneously
 * 
 * Key Insight:
 * - Small matrices don't fully utilize GPU
 * - Kernel launch overhead (~5-10 μs each)
 * - Batching amortizes overhead and increases parallelism
 */


// ============================================================================
// KERNEL 1: BATCHED GEMM - BASIC (Batch in Z dimension)
// ============================================================================
/*
 * Strategy: Use blockIdx.z for batch index
 * 
 * Grid: (ceil(N/TILE), ceil(M/TILE), batch_size)
 * Each z-slice processes one matrix pair
 * 
 * Memory Layout: Strided (all A matrices contiguous, then all B, then all C)
 *   A: [batch_size × M × K] - stride between batches = M*K
 *   B: [batch_size × K × N] - stride between batches = K*N
 *   C: [batch_size × M × N] - stride between batches = M*N
 */
__global__ void batched_gemm_basic(
        const float* __restrict__ A, // [batch_size, M, K]
        const float* __restrict__ B, // [batch_size, K, N]
        float* __restrict__ C,           // [batch_size, M, N]
        int M, int N, int K,
        int batch_size
)   {
        // Batch index from z-dimension
        int batch = blockIdx.z;

        // Bounds check
        if (batch >= batch_size) return;

        // Calculate offsets for this batch
        const float* A_batch = A + batch * M * K;
        const float* B_batch = B + batch * K * N;
        float* C_batch = C + batch * M * N;

        // Standard tiled matrix multiplication
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int tx = threadIdx.x, ty = threadIdx.y;
        int row = blockIdx.y * TILE_SIZE + ty;
        int col = blockIdx.x * TILE_SIZE + tx;

        float sum = 0.0f;
        int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

        for  (int t = 0; t < numTiles; t++) {
            int a_col = t * TILE_SIZE + tx;
            int b_row = t * TILE_SIZE + ty;

            As[ty][tx] = (row < M && a_col < K) ? A_batch[row * K + a_col] : 0.0f;
            Bs[ty][tx] = (b_row < K && col < N) ? B_batch[b_row * N + col] : 0.0f;

            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[ty][k] * Bs[k][tx];
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            C_batch[row * N + col] = sum;
        }
}


// ============================================================================
// KERNEL 2: BATCHED GEMM - REGISTER BLOCKED
// ============================================================================
/*
 * Combines batching with register blocking for maximum performance.
 * Each thread computes THREAD_TILE × THREAD_TILE elements.
 */
__global__ void batched_gemm_register_blocked(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch_size
) {          
    int batch = blockIdx.z;
    if (batch >= batch_size) return;
    
    const float* A_batch = A + batch * M * K;
    const float* B_batch = B + batch * K * N;
    float* C_batch = C + batch * M * N;

    __shared__ float As[TILE_SIZE * THREAD_TILE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE * THREAD_TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int baseRow = blockIdx.y * (TILE_SIZE * THREAD_TILE) + ty;
    int baseCol = blockIdx.x * (TILE_SIZE * THREAD_TILE) + tx;

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
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            int row = baseRow + i * TILE_SIZE;
            int col = t * TILE_SIZE + tx;
            As[ty + i * TILE_SIZE][tx] = (row < M && col < K) ?
                A_batch[row * K + col] : 0.0f;
        }

        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = t * TILE_SIZE + ty;
            int col = baseCol + j * TILE_SIZE;
            Bs[ty][tx + j * TILE_SIZE] = (row < K && col < N) ?
                B_batch[row * N + col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float a_reg[THREAD_TILE];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                a_reg[i] = As[ty + i * TILE_SIZE][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                float b_val = Bs[k][tx + j * TILE_SIZE];
                #pragma unroll
                for (int i = 0; i < THREAD_TILE; i++) {
                    sum[i][j] += a_reg[i] * b_val;
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = baseRow + i * TILE_SIZE;
            int col = baseCol + j * TILE_SIZE;
            if  (row < M && col < N) {
                C_batch[row * N + col] = sum[i][j];
            }
        }
    }
}


// ============================================================================
// KERNEL 3: BATCHED GEMM - MIXED PRECISION
// ============================================================================
/*
 * Combines batching with mixed precision (FP16 storage, FP32 accumulation)
 * Best for memory-bound scenarios with many small matrices
 */
__global__ void batched_gemm_mixed_precision(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch_size
) {
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    const half* A_batch = A + batch * M * K;
    const half* B_batch = B + batch * K * N;
    float* C_batch = C + batch * M * N;

    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A_batch[row * K + a_col] : __float2half(0.0f);
        Bs[ty][tx] = (b_row < K && col < N) ? B_batch[b_row * N + col] : __float2half(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
           sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }
    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}



// ============================================================================
// KERNEL 4: BATCHED GEMM - ARRAY OF POINTERS
// ============================================================================
/*
 * For non-contiguous batches where matrices are scattered in memory.
 * Takes arrays of pointers to each matrix.
 * 
 * This is how cuBLAS batched GEMM works (cublasSgemmBatched)
 */
__global__ void batched_gemm_pointer_array(
    const float* const* __restrict__ A_array,  // Array of pointers to A matrices
    const float* const* __restrict__ B_array,  // Array of pointers to B matrices
    float* const* __restrict__ C_array,        // Array of pointers to C matrices
    int M, int N, int K,
    int batch_size
) {
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    const float* A_batch = A_array[batch];
    const float* B_batch = B_array[batch];
    float* C_batch = C_array[batch];

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++)  {
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A_batch[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B_batch[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    } 

    if  (row < M && col < N) {
        C_batch[row * N + col] = sum;
    } 
}


// ============================================================================
// KERNEL 5: BATCHED GEMM - MIXED PRECISION + REGISTER BLOCKED (BEST)
// ============================================================================
/*
 * Maximum performance: combines all optimizations
 * - Batching (single kernel launch)
 * - Register blocking (16× work per thread)
 * - Mixed precision (2× bandwidth)
 */
__global__ void batched_gemm_mixed_register_blocked(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch_size
) {
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    const half* A_batch = A + batch * M * K;
    const half* B_batch = B + batch * K * N;
    float* C_batch = C + batch * M * N;

    __shared__ half As[TILE_SIZE * THREAD_TILE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE * THREAD_TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int baseRow = blockIdx.y * (TILE_SIZE * THREAD_TILE) + ty;
    int baseCol = blockIdx.x * (TILE_SIZE * THREAD_TILE) + tx;

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
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            int row = baseRow + i * TILE_SIZE;
            int col = t * TILE_SIZE + tx;
            As[ty + i * TILE_SIZE][tx] = (row < M && col < K) ? 
                A_batch[row * K + col] : __float2half(0.0f);
        }

        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = t * TILE_SIZE + ty;
            int col = baseCol + j * TILE_SIZE;
            Bs[ty][tx + j * TILE_SIZE] = (row < K && col < N) ? 
                B_batch[row * N + col] : __float2half(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float a_reg[THREAD_TILE];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                a_reg[i] = __half2float(As[ty + i * TILE_SIZE][k]);
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                float b_val = __half2float(Bs[k][tx + j * TILE_SIZE]);
                #pragma unroll
                for (int i = 0; i < THREAD_TILE; i++) {
                    sum[i][j] += a_reg[i] * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = baseRow + i * TILE_SIZE;
            int col = baseCol + j * TILE_SIZE;
            if (row < M && col < N) {
                C_batch[row * N + col] = sum[i][j];
            }
        }
    }
}


// ============================================================================
// BASELINE: NAIVE LOOP (Multiple kernel launches)
// ============================================================================
__global__ void gemm_single(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
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
// UTILITY FUNCTIONS
// ============================================================================

void init_matrix(float* mat, int size, float scale = 1.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
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

__global__ void convert_fp32_to_fp16(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

bool verify_batch(const float* gpu_result, const float* cpu_ref, 
                  int M, int N, int batch_size, float tolerance = 1e-2f) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < M * N; i++) {
            int idx = b * M * N + i;
            float diff = fabsf(gpu_result[idx] - cpu_ref[idx]);
            float rel = diff / (fabsf(cpu_ref[idx]) + 1e-8f);
            if (rel > tolerance && diff > tolerance) {
                printf("Mismatch at batch %d, index %d: GPU=%.6f, CPU=%.6f\n",
                       b, i, gpu_result[idx], cpu_ref[idx]);
                return false;
            }
        }
    }
    return true;
}


// ============================================================================
// BENCHMARK FUNCTION
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s M K N batch_size\n", prog);
    printf("  Computes C[b] = A[b] × B[b] for b = 0..batch_size-1\n");
    printf("  Each matrix: C(M×N) = A(M×K) × B(K×N)\n");
    printf("\nExamples:\n");
    printf("  %s 128 128 128 32     # Small matrices, batch of 32\n", prog);
    printf("  %s 256 256 256 16     # Medium matrices, batch of 16\n", prog);
    printf("  %s 64 64 64 8         # Transformer attention heads\n", prog);
    printf("  %s 512 512 512 8      # Larger matrices\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }
    
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    int batch_size = atoi(argv[4]);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║          CUDA Matrix Multiplication - Batched GEMM               ║\n");
    printf("║                           Phase 3                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Configuration:\n");
    printf("  Matrix size: C(%d×%d) = A(%d×%d) × B(%d×%d)\n", M, N, M, K, K, N);
    printf("  Batch size: %d\n", batch_size);
    printf("  Total FLOPs: %.2e (%.2e per matrix)\n", 
           2.0 * M * N * K * batch_size, 2.0 * M * N * (double)K);
    
    // Memory sizes
    size_t size_A = (size_t)M * K * batch_size;
    size_t size_B = (size_t)K * N * batch_size;
    size_t size_C = (size_t)M * N * batch_size;
    
    printf("\nMemory usage:\n");
    printf("  A: %.2f MB (%d × %d×%d)\n", size_A * 4 / 1e6, batch_size, M, K);
    printf("  B: %.2f MB (%d × %d×%d)\n", size_B * 4 / 1e6, batch_size, K, N);
    printf("  C: %.2f MB (%d × %d×%d)\n", size_C * 4 / 1e6, batch_size, M, N);
    printf("  Total: %.2f MB\n\n", (size_A + size_B + size_C) * 4 / 1e6);
    
    // Allocate host memory
    float* h_A = (float*)malloc(size_A * sizeof(float));
    float* h_B = (float*)malloc(size_B * sizeof(float));
    float* h_C = (float*)malloc(size_C * sizeof(float));
    float* h_C_ref = (float*)malloc(size_C * sizeof(float));
    
    // Initialize
    srand(42);
    init_matrix(h_A, size_A);
    init_matrix(h_B, size_B);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    half *d_A_fp16, *d_B_fp16;
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_fp16, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, size_B * sizeof(half)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));
    
    // Convert to FP16
    dim3 conv_block(256);
    dim3 conv_grid_A((size_A + 255) / 256);
    dim3 conv_grid_B((size_B + 255) / 256);
    convert_fp32_to_fp16<<<conv_grid_A, conv_block>>>(d_A, d_A_fp16, size_A);
    convert_fp32_to_fp16<<<conv_grid_B, conv_block>>>(d_B, d_B_fp16, size_B);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Grid configurations
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid_basic((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE,
                    batch_size);
    
    int block_output = TILE_SIZE * THREAD_TILE;
    dim3 grid_reg((N + block_output - 1) / block_output,
                  (M + block_output - 1) / block_output,
                  batch_size);
    
    dim3 grid_single((N + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid configurations:\n");
    printf("  Basic batched:    Grid(%d, %d, %d)\n", grid_basic.x, grid_basic.y, grid_basic.z);
    printf("  Register blocked: Grid(%d, %d, %d)\n", grid_reg.x, grid_reg.y, grid_reg.z);
    printf("  Single (loop):    Grid(%d, %d) × %d launches\n\n", 
           grid_single.x, grid_single.y, batch_size);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int warmup = 3;
    int runs = 10;
    float ms;
    double flops = 2.0 * M * N * K * batch_size;
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("                       PERFORMANCE BENCHMARK                        \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    // Benchmark 1: Naive loop (multiple kernel launches)
    for (int i = 0; i < warmup; i++) {
        for (int b = 0; b < batch_size; b++) {
            gemm_single<<<grid_single, block>>>(
                d_A + b * M * K, d_B + b * K * N, d_C + b * M * N, M, N, K);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        for (int b = 0; b < batch_size; b++) {
            gemm_single<<<grid_single, block>>>(
                d_A + b * M * K, d_B + b * K * N, d_C + b * M * N, M, N, K);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Naive Loop (%d launches)        : %8.3f ms, %8.2f GFLOPS\n", 
           batch_size, ms/runs, (flops / (ms/runs/1000)) / 1e9);
    float gflops_naive = (flops / (ms/runs/1000)) / 1e9;
    
    // Benchmark 2: Batched basic
    for (int i = 0; i < warmup; i++) {
        batched_gemm_basic<<<grid_basic, block>>>(d_A, d_B, d_C, M, N, K, batch_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        batched_gemm_basic<<<grid_basic, block>>>(d_A, d_B, d_C, M, N, K, batch_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Batched Basic (1 launch)        : %8.3f ms, %8.2f GFLOPS\n", 
           ms/runs, (flops / (ms/runs/1000)) / 1e9);
    float gflops_basic = (flops / (ms/runs/1000)) / 1e9;
    
    // Benchmark 3: Batched register blocked
    for (int i = 0; i < warmup; i++) {
        batched_gemm_register_blocked<<<grid_reg, block>>>(d_A, d_B, d_C, M, N, K, batch_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        batched_gemm_register_blocked<<<grid_reg, block>>>(d_A, d_B, d_C, M, N, K, batch_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Batched + Register Blocked      : %8.3f ms, %8.2f GFLOPS\n", 
           ms/runs, (flops / (ms/runs/1000)) / 1e9);
    float gflops_reg = (flops / (ms/runs/1000)) / 1e9;
    
    // Benchmark 4: Batched mixed precision
    for (int i = 0; i < warmup; i++) {
        batched_gemm_mixed_precision<<<grid_basic, block>>>(d_A_fp16, d_B_fp16, d_C, M, N, K, batch_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        batched_gemm_mixed_precision<<<grid_basic, block>>>(d_A_fp16, d_B_fp16, d_C, M, N, K, batch_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Batched + Mixed Precision       : %8.3f ms, %8.2f GFLOPS\n", 
           ms/runs, (flops / (ms/runs/1000)) / 1e9);
    float gflops_mixed = (flops / (ms/runs/1000)) / 1e9;
    
    // Benchmark 5: Batched mixed + register blocked (BEST)
    for (int i = 0; i < warmup; i++) {
        batched_gemm_mixed_register_blocked<<<grid_reg, block>>>(d_A_fp16, d_B_fp16, d_C, M, N, K, batch_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) {
        batched_gemm_mixed_register_blocked<<<grid_reg, block>>>(d_A_fp16, d_B_fp16, d_C, M, N, K, batch_size);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Batched + Mixed + RegBlock      : %8.3f ms, %8.2f GFLOPS\n", 
           ms/runs, (flops / (ms/runs/1000)) / 1e9);
    float gflops_best = (flops / (ms/runs/1000)) / 1e9;
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("                           SUMMARY                                  \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("Speedup vs Naive Loop:\n");
    printf("  Batched Basic:           %.2fx\n", gflops_basic / gflops_naive);
    printf("  Batched + RegBlock:      %.2fx\n", gflops_reg / gflops_naive);
    printf("  Batched + Mixed:         %.2fx\n", gflops_mixed / gflops_naive);
    printf("  Batched + Mixed + Reg:   %.2fx\n", gflops_best / gflops_naive);
    
    printf("\nKernel launch overhead eliminated: %d launches → 1 launch\n", batch_size);
    
    // Verification
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("                         VERIFICATION                               \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    if (M * N * batch_size < 10000000) {  // Only verify for reasonable sizes
        printf("Computing CPU reference...\n");
        for (int b = 0; b < batch_size; b++) {
            matmul_cpu(h_A + b * M * K, h_B + b * K * N, h_C_ref + b * M * N, M, N, K);
        }
        
        // Verify batched register blocked
        batched_gemm_register_blocked<<<grid_reg, block>>>(d_A, d_B, d_C, M, N, K, batch_size);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
        
        if (verify_batch(h_C, h_C_ref, M, N, batch_size)) {
            printf(" Batched Register Blocked: PASSED\n");
        } else {
            printf(" Batched Register Blocked: FAILED\n");
        }
        
        printf("\nSample output (batch 0):\n");
        printf("  C[0][0,0] = %.6f (CPU: %.6f)\n", h_C[0], h_C_ref[0]);
    } else {
        printf("Skipping verification (matrices too large)\n");
    }
    
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("                    NEURAL NETWORK CONTEXT                          \n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    printf("This batched GEMM is used in:\n");
    printf("  • Batch processing: %d samples × (%d×%d) layer\n", batch_size, M, N);
    printf("  • Transformer attention: %d heads × (%d×%d) Q×K^T\n", batch_size, M, N);
    printf("  • Multi-head self-attention parallelism\n");
    printf("  • Recurrent networks (LSTM/GRU gates)\n");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("\n Complete!\n\n");
    
    return 0;
}
    

























    














