//
// Created by yuzhang on 2022/4/18.
//

#include <stdio.h>
#include <assert.h>

#define N 100000
#define FULL_MASK unsigned(0xffffffff)
#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define BLOCK_NUM ((N+BLOCK_SIZE-1)/BLOCK_SIZE)
#define MAX_WARP_NUM_PER_BLOCK 32


__device__ int block_counter_0 = 0;
__device__ int block_counter_1 = 0;
__device__ int block_prefix_sum;


__global__
void init(int *a) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    a[i] = 1;
  }
  if(i == 0) {
    block_counter_0 = block_counter_1 = 0;
  }
}

__global__
void naive_scan(int *a, int *sum) {
  sum[0] = a[0];
  int i = 1;
  while(i < N) {
    sum[i] = sum[i-1] + a[i];
    ++i;
  }
}

__global__
void check_and_print_res(int *a, int *sum) {
  for(int i = 0; i < N; ++i) {
    // printf("%8d %8d %8d\n", i, a[i], sum[i]);
    assert(a[i] == sum[i]);
  }
}



__global__  void parallel_scan_inplace(int *d_data, int *block_sum) {

  // real_block_id is real execution order of blocks
  __shared__ int real_block_id;
  if (threadIdx.x == 0) {
    real_block_id = atomicAdd(&block_counter_0, 1);
  }
  __syncthreads();

  if(threadIdx.x + real_block_id * blockDim.x >= N) return;

  int gtid = threadIdx.x + real_block_id * blockDim.x;
  int ltid = threadIdx.x;
  int lane_id = ltid % WARP_SIZE;
  int warp_id = ltid / WARP_SIZE;
  int warp_num = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE; // warp_num should not be greater than MAX_WARP_NUM_PER_BLOCK


  __shared__ int warp_sum[MAX_WARP_NUM_PER_BLOCK];

  int val = d_data[gtid], temp0;
  // warp inner scan
  for(int d=1; d<WARP_SIZE; d<<=1) {
    temp0 = __shfl_up_sync(FULL_MASK, val, d);
    if (lane_id >= d) val += temp0;
  }

  __syncthreads();

  // add sum of a whole warp into a position
  if(lane_id == WARP_SIZE-1) warp_sum[warp_id] = val;

  __syncthreads();

  // We assume warp num < 32, thus we can use the first warp to calculate warp-level prefix sum.
  if(ltid < WARP_SIZE) {

    temp0 = 0;
    if(ltid < warp_num)
      temp0 = warp_sum[ltid];

    for (int d=1; d<warp_num; d<<=1) {
      int temp1 = __shfl_up_sync(FULL_MASK, temp0, d);
      if(lane_id >= d) temp0 += temp1;
    }

    // write back
    if(ltid < warp_num) warp_sum[ltid] = temp0;

  }

  __syncthreads();


  // add sum of previous all warps
  if(ltid >= WARP_SIZE) val += warp_sum[warp_id - 1];

  d_data[gtid] = val;
  if(ltid == BLOCK_SIZE-1) {
    block_sum[real_block_id + 1] = val;
  }

  // ensure sync in a block
  __syncthreads();


  if (ltid == 0) {
    // [Lock Begin]
    // do-nothing atomic forces a load each time
    while(atomicAdd(&block_counter_1, 0) != real_block_id);

    __threadfence();
    if(real_block_id == 0) block_sum[real_block_id] = 0;
    else block_sum[real_block_id] += block_sum[real_block_id-1];

    __threadfence(); // wait for write completion
    atomicAdd(&block_counter_1, 1); // faster than plain addition
    // [Lock End]
  }

  __syncthreads();

  // write back
  d_data[gtid] += block_sum[real_block_id];
}


int main() {

  assert((BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE <= MAX_WARP_NUM_PER_BLOCK);

  int *da, *sum, *block_sum;
  cudaMalloc((void **)&da, N*sizeof(int));
  cudaMalloc((void **)&sum, N*sizeof(int));
  cudaMalloc((void **)&block_sum, BLOCK_NUM*sizeof(int));


  for(int i = 0; i < 10; ++i) {
    init<<<BLOCK_NUM, BLOCK_SIZE>>>(da);
    naive_scan<<<1, 1>>>(da, sum);
    parallel_scan_inplace<<<BLOCK_NUM, BLOCK_SIZE>>>(da, block_sum);
    check_and_print_res<<<1, 1>>>(da, sum);
    cudaDeviceSynchronize(); // make printf() in kernel works
  }

  cudaFree(da);
  cudaFree(sum);
  cudaFree(block_sum);

  return 0;
}