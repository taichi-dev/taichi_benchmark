#include "scan.cuh"
using namespace cub;
CachingDeviceAllocator  g_allocator(true);

int main(int argc, char** argv) {
    int num_items = 4096;
    if(argc > 1) num_items = std::atoi(argv[1]);
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *h_in = new float [num_items];
    float *h_reference = new float[num_items];
    float *h_out = new float [num_items + 1];
    Initialize(h_in, num_items);
    Solve(h_in, h_reference, num_items);

    cudaMalloc(&d_in, num_items * sizeof(float));
    cudaMalloc(&d_out, (num_items + 1) * sizeof(float));
    cuErrCheck(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    CubDebugExit(DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
    cudaDeviceSynchronize();
    cuErrCheck(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f", milliseconds);

    cuErrCheck(cudaMemcpy(h_out, d_out, sizeof(float) * (num_items), cudaMemcpyDeviceToHost));
    TestResult(h_out, h_reference, num_items);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_reference;
    delete[] h_in;
    delete[] h_out;
    return 0;
}