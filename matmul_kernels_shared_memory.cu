#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// VERSION: REGISTER BLOCKING (Each thread computes multiple elements)
// ============================================================================
#define TILE_SIZE 16
#define THREAD_TILE 4  // Each thread computes 4×4 elements

__global__ void matmul_register_blocked(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    __shared__ float As[TILE_SIZE * THREAD_TILE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE * THREAD_TILE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Calculate starting position for this thread's output tile
    int row_base = by * TILE_SIZE * THREAD_TILE + ty;
    int col_base = bx * TILE_SIZE * THREAD_TILE + tx;
    
    // Accumulator registers for THREAD_TILE × THREAD_TILE elements
    float sum[THREAD_TILE][THREAD_TILE];
    for (int i = 0; i < THREAD_TILE; i++) {
        for (int j = 0; j < THREAD_TILE; j++) {
            sum[i][j] = 0.0f;
        }
    }

    // Loop over K dimension in tiles
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Collaboratively load tiles into shared memory
        // Each thread loads THREAD_TILE elements from A
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            int row = row_base + i * TILE_SIZE;
            int col = t * TILE_SIZE + tx;
            if (row < M && col < K) {
                As[ty + i * TILE_SIZE][tx] = A[row * K + col];
            } else {
                As[ty + i * TILE_SIZE][tx] = 0.0f;
            }
        }

        // Each thread loads THREAD_TILE elements from B
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = t * TILE_SIZE + ty;
            int col = col_base + j * TILE_SIZE;
            if (row < K && col < N) {
                Bs[ty][tx + j * TILE_SIZE] = B[row * N + col];
            } else {
                Bs[ty][tx + j * TILE_SIZE] = 0.0f;
            }
        }
            
        __syncthreads();
        
        // Compute THREAD_TILE × THREAD_TILE partial products
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++) {
                    sum[i][j] += As[ty + i * TILE_SIZE][k] * 
                                 Bs[k][tx + j * TILE_SIZE];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int row = row_base + i * TILE_SIZE;
            int col = col_base + j * TILE_SIZE;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}

// ============================================================================
// UTILITIES
// ============================================================================
void initialize_matrix(float* M, int rows, int cols, bool random) {
    for (int i = 0; i < rows * cols; i++) {
        if (random) {
            M[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        } else {
            M[i] = 0.0f;
        }
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main(int argc, char** argv) {
    // Matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    
    printf("Matrix Multiplication: C(%d,%d) = A(%d,%d) * B(%d,%d)\n\n", M, N, M, K, K, N);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    // Initialize matrices
    srand(42);
    initialize_matrix(h_A, M, K, true);
    initialize_matrix(h_B, K, N, true);
    initialize_matrix(h_C, M, N, false);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Setup execution configuration
    // Grid must account for THREAD_TILE
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE * THREAD_TILE - 1) / (TILE_SIZE * THREAD_TILE),
              (M + TILE_SIZE * THREAD_TILE - 1) / (TILE_SIZE * THREAD_TILE));
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);
    printf("Each thread computes: %d×%d elements\n\n", THREAD_TILE, THREAD_TILE);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up - CORRECT FUNCTION NAME
    matmul_register_blocked<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    int num_iterations = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        matmul_register_blocked<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // Calculate Performance
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    double avg_time = milliseconds / num_iterations;
    
    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;
    double gflops = (flops / avg_time) / 1e6;

    printf("Register-Blocked Matrix Multiplication\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("Execution Time: %.3f ms\n", avg_time);
    printf("Performance:    %.2f GFLOPS\n", gflops);
    printf("════════════════════════════════════════════════════════════\n");
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    printf("\nFirst element of result: C[0][0] = %.6f\n", h_C[0]);
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("\n✓ Computation complete!\n");
    
    return 0;
}