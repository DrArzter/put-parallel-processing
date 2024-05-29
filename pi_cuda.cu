#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void count_points(int *count, int N, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        double x = curand_uniform(&state);
        double y = curand_uniform(&state);
        if (x * x + y * y <= 1.0) {
            atomicAdd(count, 1);
        }
    }
}

int main() {
    const int N = 1000000;
    int count = 0;
    int *d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    count_points<<<blocksPerGrid, threadsPerBlock>>>(d_count, N, time(0));

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    double pi = 4.0 * count / N;
    std::cout << "Approximate value of pi: " << pi << std::endl;
    return 0;
}
