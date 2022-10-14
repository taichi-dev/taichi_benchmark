#include "reduce_sum.cuh"

__device__ float WarpReduceSum(float val) {
    int lane = threadIdx.x & 31;
    float tmp = __shfl_up_sync(0xffffffff, val, 1);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 2);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 4);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 8);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up_sync(0xffffffff, val, 16);
    if (lane >= 16) val += tmp;
    __syncthreads();
    return val;
}

__device__ float BlockReduceSum(float val) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    __shared__ float warp_sum[32];

    val = WarpReduceSum(val);
    if(lane == 31) warp_sum[warp_id] = val;
    __syncthreads();
    if(warp_id == 0) {
        if (lane >= 1) warp_sum[lane] += warp_sum[lane-1];
        __syncwarp();
        if (lane >= 2) warp_sum[lane] += warp_sum[lane-2];
        __syncwarp();
        if (lane >= 4) warp_sum[lane] += warp_sum[lane-4];
        __syncwarp();
        if (lane >= 8) warp_sum[lane] += warp_sum[lane-8];
        __syncwarp();
        if (lane >= 16) warp_sum[lane] += warp_sum[lane-16];
    }
    __syncthreads();
    if(warp_id > 0) val += warp_sum[warp_id-1];
    return val;
}

__global__
void ReduceSumKernel(float *in, float *out,
    int num_items, int num_part) {
    float val = 0;
    int idx = 0;
    for(int ii = blockIdx.x; ii < num_part; ii += gridDim.x) {
        idx = threadIdx.x +  blockDim.x * ii;
        val += idx < num_items ? in[idx] : 0;
    }
    val = BlockReduceSum(val);
    if(threadIdx.x == blockDim.x - 1&& (idx < num_items) || (idx == num_items - 1)) {
        out[blockIdx.x] = val;
    }
}

void ReduceSum(float *d_in, float *d_out, int num_items) {
    int TPB = TPB1D;
    int num_part = (num_items + TPB - 1) / TPB;
    int BPG = std::min<int>(num_part, 256);
    ReduceSumKernel<<<BPG, TPB>>> (
        d_in, d_out, num_items, num_part);
    ReduceSumKernel<<<1, TPB>>>(
        d_out, d_out, BPG, 1);
}

int main(int argc, char **argv) {
    int num_items = 4096;
    if(argc > 1) num_items = std::atoi(argv[1]);
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *h_in = new float [num_items];
    float *h_reference = new float;
    float *h_out = new float;
    Initialize(h_in, num_items);
    Solve(h_in, h_reference, num_items);

    cudaMalloc(&d_in, num_items * sizeof(float));
    // Loose boundary
    cudaMalloc(&d_out, (num_items + TPB1D - 1) / TPB1D * 2 * sizeof(float));
    cudaMemset(d_out, 0, (num_items + TPB1D - 1) / TPB1D * 2 * sizeof(float));
    cuErrCheck(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    ReduceSum(d_in, d_out, num_items);

    cudaDeviceSynchronize();
    cuErrCheck(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f", milliseconds);

    cuErrCheck(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    TestResult(h_out, h_reference);

    cudaFree(d_in);
    cudaFree(d_out);
    delete h_reference;
    delete h_out;
    delete[] h_in;

    return 0;
}