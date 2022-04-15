#include "scan.cuh"

int main(int argc, char **argv) {
    int num_items = 4096;
    if(argc > 1) num_items = std::atoi(argv[1]);
    float *h_in = new float [num_items];
    float *h_reference = new float[num_items];
    float *h_out = new float [num_items + 1];
    thrust::device_vector<float> u(num_items);
    Initialize(h_in, num_items);
    Solve(h_in, h_reference, num_items);

    cuErrCheck(cudaMemcpy(thrust::raw_pointer_cast(&u[0]), h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    thrust::inclusive_scan(u.begin(), u.end(), u.begin());
    cudaDeviceSynchronize();
    cuErrCheck(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f", milliseconds);
    cuErrCheck(cudaMemcpy(h_out, thrust::raw_pointer_cast(&u[0]), sizeof(float) * (num_items), cudaMemcpyDeviceToHost));
    TestResult(h_out, h_reference, num_items);

    delete[] h_in;
    delete[] h_out;
    delete[] h_reference;
}
