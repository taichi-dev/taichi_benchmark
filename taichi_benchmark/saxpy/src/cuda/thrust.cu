#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/transform.h>

#include "timer.h"

using namespace thrust::placeholders;

__host__ static __inline__ float rand_01()
{
    return ((float)rand()/RAND_MAX);
}

template<int nesting_factor>
int saxpy(int _N) {
    int N = _N * _N;
    thrust::host_vector<float> x(N), y(N);
    thrust::generate(x.begin(), x.end(), rand_01);
    thrust::generate(y.begin(), y.end(), rand_01);
    thrust::device_vector<float> d_x = x; // alloc and copy host to device
    thrust::device_vector<float> d_y = y;

    Timer tmr;
    tmr.start();
    size_t nIter = 5000;
    for (int i = 0; i < nIter; ++i) {
        // Perform SAXPY on 1M elements
        if (nesting_factor == 1) { 
            thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), 2.0f * _1 + _2);
        } else if (nesting_factor == 2) {
            thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), 2.0f * (2.0f * _1 + _2) + _2);
        } else if (nesting_factor == 4) {
            thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), 2.0f * (2.0f * (2.0f * (2.0f * _1 + _2) + _2) + _2) + _2);
        } else if (nesting_factor == 8) {
            thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), 2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * _1 + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2);
        } else if (nesting_factor == 16) {
            thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_y.begin(), 2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * (2.0f * _1 + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2) + _2);
        } 
    }
    tmr.stop();
    double avg_time = tmr.getTimeMillisecond() / nIter;
    double GFlops = 1e-6 * N * 2 * nesting_factor / avg_time;
    double GBs = 1e-6 * N * sizeof(float) * 3 / avg_time;
#ifdef JSON_OUTPUT
    printf("{\"N\": %d, \"fold\":%d, \"time\":%.3lf, \"gflops\":%.3lf, \"gbs\": %.3lf}\n",  _N, nesting_factor, avg_time, GFlops, GBs);
#else
    printf("%dx%d@%d, %.3lf ms, %.3lf GFLOPS, %.3lf GB/s\n", _N, _N, nesting_factor, avg_time, GFlops, GBs);
#endif
    
    y = d_y; // copy results to the host vector

    return 0;
}

int main() {
    int N = 256;
    for(int i = 0; i < 5; ++i) {
        saxpy<1>(N);
        saxpy<2>(N);
        saxpy<4>(N);
        saxpy<8>(N);
        saxpy<16>(N);
        N *= 2;
    }
    return 0;
}
