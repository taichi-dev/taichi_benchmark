// Parallel Prefix Sum (Scan)
// Ref[0]: https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
// Ref[1]: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu

// Last update: July 12, 2022
// Author: Bo Qiao

#include<iostream>
#include<cstdlib>
#include<cmath>
#include<vector>
#include<limits>

#include<thrust/scan.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/copy.h>

#define TYPE float

// Scan using shuffle instructions
__global__ void shfl_scan(TYPE *data, int len, TYPE *partial_sums = NULL) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x; 
  if (idx < len) {
    extern __shared__ TYPE sums[];
    const int warp_sz = 32;
    int lane_id = idx % warp_sz;
    int warp_id = threadIdx.x / warp_sz;
    
    TYPE value = data[idx];

    // intra-warp scan
    for (int i=1; i<=32; i*=2) {
      TYPE n = __shfl_up_sync(0xffffffff, value, i);
      if (lane_id >= i) {
        value += n; 
      }
    }
    
    // put warp scan results to smem
    if (threadIdx.x % warp_sz == warp_sz - 1) {
      sums[warp_id] = value;
    }
    __syncthreads();

    // inter-warp scan, use the first thread in the first warp
    if (warp_id == 0 && lane_id == 0) {
      for (int i=1; i<blockDim.x / warp_sz; i++) {
        sums[i] += sums[i-1];
      }
    }
    __syncthreads();
    
    // update data with warp_sums
    TYPE warp_sum = 0;
    if (warp_id > 0) {
      warp_sum = sums[warp_id - 1];
    }
    value += warp_sum;
    data[idx] = value;

    // update partial sums if applicable
    if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
      partial_sums[blockIdx.x] = value;
    }
  }
}

// Aux function uniform add
__global__ void uniform_add(TYPE *data, TYPE *partial_sums, int len) {
  __shared__ TYPE buf;
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx <= len) {
    if (threadIdx.x == 0) {
      buf = partial_sums[blockIdx.x];
    }
    __syncthreads();
    data[idx] += buf;
  }
}

bool AreSameFloats(float a, float b)
{
  return fabs(a - b) < std::numeric_limits<float>::epsilon();
}

// CPU golden results
// work-efficient sequential scan: exactly n additions, O(n)
bool compare_scan_golden(TYPE* output, TYPE* input, int len) {
  bool pass = true;
  TYPE sum = 0;
  for (int j=0; j<len; j++) {
    sum += input[j];
    if (!AreSameFloats(sum, output[j])) {
      pass = false; 
      std::cout << "[Fail] At pos " << j << ", golden " << sum << ", output " << output[j] << std::endl;
      break;
    }
  }
  return pass;
}

int main(int argc, char **argv) {
  // Parse input size
  int n_elements = 100000;
  if (argc > 1) {
    n_elements = atoi(argv[1]);
  }
  std::cout << "[Info] Number of elements: " << n_elements << std::endl;

  const int blockSize = 256;
  std::cout << "[Info] Block Size: " << blockSize << std::endl;
  int shmem_sz = blockSize/32 * sizeof(TYPE);

  // Buffer allocations
  TYPE *h_data, *h_result, *h_result_golden;
  cudaMallocHost(reinterpret_cast<void **>(&h_data), sizeof(TYPE) * n_elements);
  cudaMallocHost(reinterpret_cast<void **>(&h_result), sizeof(TYPE) * n_elements);
  cudaMallocHost(reinterpret_cast<void **>(&h_result_golden), sizeof(TYPE) * n_elements);

  // Initialize host data
  for (size_t i=0; i<n_elements; i++) {
      h_data[i] = 1.0;
      h_result_golden[i] = 0.0;
  }

  // Allocate device buffers to hold input and all intermediate partial sums
  std::vector<TYPE*> device_data;
  std::vector<int> device_data_ele_sz;
  for (int ne=n_elements; ne>1; ne=(ne+blockSize-1)/blockSize) {
    TYPE *data;
    cudaMalloc(reinterpret_cast<void **>(&data), sizeof(TYPE) * ne);
    cudaMemset(data, 0, sizeof(TYPE) * ne);
    device_data.push_back(data);
    device_data_ele_sz.push_back(ne);
  }

  // Copy data to device
  cudaMemcpy(device_data[0], h_data, sizeof(TYPE) * n_elements, cudaMemcpyHostToDevice);

  // Use events to time device execution
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Kernel launch
  cudaEventRecord(start, 0);
  int it_id = 0;
  for (int ne=n_elements; ne>1; ne=(ne+blockSize-1)/blockSize) {
    int grid_size = (ne+blockSize-1)/blockSize;
    if (grid_size == 1) {
      shfl_scan<<<1, blockSize, shmem_sz>>>(device_data.back(), ne);
    } else {
      shfl_scan<<<grid_size, blockSize, shmem_sz>>>(device_data[it_id], ne,
		      device_data[it_id+1]);
    }
    it_id++;
  }
  
  // Add partial sums back
  for (int i=device_data_ele_sz.size()-2; i>=0; i--) {
    int p_grid_sz = (device_data_ele_sz[i]+blockSize-1)/blockSize - 1;
    uniform_add<<<p_grid_sz, blockSize>>>(device_data[i]+blockSize,
		    device_data[i+1], device_data_ele_sz[i]);
  }
  
  cudaEventRecord(stop, 0);

  // Copy result back to host
  cudaMemcpy(h_result, device_data[0], sizeof(TYPE) * n_elements,
		  cudaMemcpyDeviceToHost);

  // Compare ground truth
  compare_scan_golden(h_result, h_data, n_elements);

  float time_ms = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  std::cout << "[Benchmark] Execution time: " << time_ms << " ms" << std::endl;
  std::cout << "[Benchmark] Achieved bandwidth: " << 
	  (sizeof(TYPE) * (float)n_elements / 1000000.0) / time_ms << " GB/s" << std::endl;

  // Thrust in-place scan as comparison
  thrust::device_vector<TYPE> d_values(h_data, h_data + n_elements);

  cudaEventRecord(start, 0);
  thrust::inclusive_scan(d_values.begin(), d_values.end(), d_values.begin());
  cudaEventRecord(stop, 0);

  thrust::host_vector<TYPE> h_thrust_result(d_values.begin(), d_values.end());
  compare_scan_golden(&h_thrust_result[0], h_data, n_elements);

  cudaEventSynchronize(stop);
  time_ms = 0;
  cudaEventElapsedTime(&time_ms, start, stop);
  std::cout << "[Benchmark] Thrust Execution time: " << time_ms << " ms" << std::endl;
  std::cout << "[Benchmark] Thrust Achieved bandwidth: " << 
	  (sizeof(TYPE) * (float)n_elements / 1000000.0) / time_ms << " GB/s" << std::endl;

  // Clean ups
  cudaFreeHost(h_data); 
  cudaFreeHost(h_result); 
  cudaFreeHost(h_result_golden); 
  for (auto d_data: device_data) {
    cudaFree(d_data); 
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

