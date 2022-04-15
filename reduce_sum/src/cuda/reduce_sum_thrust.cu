#include "reduce_sum.cuh"

int main(int argc, char **argv) {
    int num_items = 4096;
    if(argc > 1) num_items = std::atoi(argv[1]);
    float *h_in = new float [num_items];
    float *h_reference = new float;
    thrust::device_vector<float> u(num_items);
    Initialize(h_in, num_items);
    Solve(h_in, h_reference, num_items);

    cuErrCheck(cudaMemcpy(thrust::raw_pointer_cast(&u[0]), h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float h_out = thrust::reduce(u.begin(), u.end(), 0.0f, thrust::plus<float>());
    cudaDeviceSynchronize();
    cuErrCheck(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f", milliseconds);
    TestResult(&h_out, h_reference);

    delete[] h_in;
    delete[] h_reference;
}
