#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call) \
    do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                            cudaGetErrorString(err)); \
                exit(EXIT_FAILURE); \
        } \
    } while(0)


// ============================================================================
// VERSION 3: SHARED MEMORY (Cache tiles in shared memory)
// ============================================================================
// Increased from 16 to 32
#define TILE_SIZE 32  

__global__ void matmul_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if ((t * TILE_SIZE + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial product
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
    int M = 1024;  // Rows of A and C
    int N = 1024;  // Cols of B and C
    int K = 1024;  // Cols of A, Rows of B
    
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
    dim3 block(32, 32); // 16×16 = 256 threads per block
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid: (%d, %d), Block: (%d, %d)\n\n", grid.x, grid.y, block.x, block.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up
    matmul_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    int num_iterations = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        matmul_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

    printf("Shared Memory Tiled Matrix Multiplication\n");
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






