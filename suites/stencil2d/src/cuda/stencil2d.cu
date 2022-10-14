#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <chrono>

#define BLOCK_SIZE 128

void init(float* buf, int N) {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            float* ptr = buf + y * N + x;                
            if (x == 0 || x == N - 1) {
                *ptr = 1.0;
            } else {
                *ptr = 0.0;
            }
        }
    }
}

__global__ void jacobi_step(const float* xbuf, float* ybuf, int N) {
    int nIter = (N * N + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    for (int iter = 0; iter < nIter; ++iter) {
        int globalIdx = blockIdx.x * blockDim.x + threadIdx.x + iter * blockDim.x * gridDim.x;
        if (globalIdx < N * N) {
            int coord_x = globalIdx % N;
            int coord_y = globalIdx / N;
            if ((coord_x == 0) || (coord_x == N - 1) || (coord_y == 0) || (coord_y == N - 1)) {
                ybuf[globalIdx] = xbuf[globalIdx];
            } else {
                float x = 0.f;
                x += xbuf[(coord_y - 1) * N + coord_x];
                x += xbuf[(coord_y + 1) * N + coord_x];
                x += xbuf[coord_y * N + coord_x - 1];
                x += xbuf[coord_y * N + coord_x + 1];
                x *= 0.25;
                ybuf[globalIdx] = x;
            }
        }
    }
}

void benchmark(int N, int nIter) {
    int GRID_SIZE = N * N / BLOCK_SIZE;
    float* buf = NULL;
    float* xbuf = NULL;
    float* ybuf = NULL;
    cudaMallocHost(&buf, sizeof(float) * N * N);
    cudaMalloc(&xbuf, sizeof(float) * N * N);
    cudaMalloc(&ybuf, sizeof(float) * N * N);
    init(buf, N);
    cudaMemcpy(xbuf, buf, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    typedef std::chrono::high_resolution_clock Clock;

    auto st = Clock::now();
    for (int i = 0; i < nIter; ++i) {
        jacobi_step<<<GRID_SIZE, BLOCK_SIZE>>>(xbuf, ybuf, N);
        jacobi_step<<<GRID_SIZE, BLOCK_SIZE>>>(ybuf, xbuf, N);
    }
    auto et = Clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    double avg_time = duration_us / nIter / 1000.0 / 2.0;
    double Gflops = 1e-6 * N * N * (4 + 1) / avg_time;
    double GBs = 1e-6 * 2 * N * N * 4 / avg_time;
#ifdef JSON_OUTPUT
    printf("{\"N\":%d,\"time\":%.3lf,\"gflops\":%.3lf,\"gbs\":%.3lf}\n", N, avg_time, Gflops, GBs);
#else
    printf("Stencil %dx%d, %.3lfGFLOPS, %.3lfGB/s\n", N, N, Gflops, GBs);
#endif
    cudaMemcpy(buf, xbuf, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    cudaFree(buf);
    cudaFree(xbuf);
    cudaFree(ybuf);
    
}

int main(int argc, char* argv[]) {
    int N = 64;
    if (argc >= 2) {
        N = atoi(argv[1]);
    }
    benchmark(N, 10000);
    return 0;
}
