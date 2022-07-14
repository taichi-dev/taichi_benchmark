#include "scan.cuh"

__device__ float WarpScan(float val) {
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

__device__ float BlockScan(float val) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    __shared__ float warp_sum[32];

    val = WarpScan(val);
    __syncthreads();
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
    __syncthreads();
    return val;
}

__global__ 
void ScanKernel(float *in, float *out,
        float *buffer, int num_items, int num_part) {
    for(int ii = blockIdx.x; ii < num_part; ii += gridDim.x) {
        int idx = blockDim.x * ii + threadIdx.x;
        float val = idx < num_items ? in[idx] : 0;
        val = BlockScan(val);
        if(idx < num_items) out[idx] = val;
        if(threadIdx.x == blockDim.x - 1 && idx < num_items) {
            buffer[ii] = val;
        }
    }
}

__global__ void AddBaseKernel(float *buffer, float *out,
    int num_items, int num_part) {
    for(int ii = blockIdx.x; ii < num_part; ii += gridDim.x) {
        if(ii == 0) continue;
        int idx = ii * blockDim.x + threadIdx.x;
        if(idx < num_items) out[idx] += buffer[ii - 1];
    }
}

void Scan(float *d_in, float *d_out, float *buffer, int num_items) {
    int TPB = TPB1D;
    int num_part = (num_items + TPB - 1) / TPB;
    int BPG = std::min<int>(num_part, 256);
    ScanKernel<<<BPG, TPB>>> (
        d_in, d_out, buffer, num_items, num_part);
    if(num_part >= 2) {
        Scan(buffer, buffer + num_part, buffer, num_part);
        AddBaseKernel<<<BPG, TPB>>>(buffer+num_part, d_out, num_items, num_part);
    }
}

int main(int argc, char **argv) {
    int num_items = 4096;
    if(argc > 1) num_items = std::atoi(argv[1]);
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *buffer = nullptr;
    float *h_in = new float [num_items];
    float *h_out = new float [num_items];
    float *h_reference = new float [num_items];

    Initialize(h_in, num_items);
    Solve(h_in, h_reference, num_items);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_in, num_items * sizeof(float));
    cudaMalloc(&d_out, num_items * sizeof(float));
    // Loose array
    cudaMalloc(&buffer, (num_items + TPB1D - 1) / TPB1D * 4 * sizeof(float));
    cudaMemset((void *)buffer, 0, (num_items + TPB1D - 1) / TPB1D * 4 * sizeof(float));
    
    cuErrCheck(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));
    
    cudaEventRecord(start);

    Scan(d_in, d_out, buffer, num_items);
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
    cudaFree(buffer);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_reference;
    delete[] h_out;
    return 0;
}