#pragma once

#include <curand_kernel.h>

//__device__ curandState randState;
//
//__device__ void initRandom(unsigned int seed)
//{
//    curand_init(seed, 0, 0, &randState);
//}
//
//// Generate a uniform distributed random number[0, 1]
//__device__ float getRandom()
//{
//    curand_uniform(&randState);
//}