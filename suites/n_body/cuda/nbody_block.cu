#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct {
    float x;
    float y;
    float z;
    float w;
} Body;

typedef struct { Body *pos, *vel; } BodySystem;


void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(Body *p, Body *v, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
      for (int j = 0; j < BLOCK_SIZE; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }
      __syncthreads();
    }

    v[i].x += dt*Fx; 
    v[i].y += dt*Fy; 
    v[i].z += dt*Fz;
    //v[i].x = atomicAdd(&v[i].x, dt*Fx);
    //v[i].y = atomicAdd(&v[i].y, dt*Fy);
    //v[i].z = atomicAdd(&v[i].z, dt*Fz);
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);
  
  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  
  int bytes = 2*nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  BodySystem p = { (Body*)buf, ((Body*)buf) + nBodies };

  randomizeBodies(buf, 8*nBodies); // Init pos / vel data

  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  BodySystem d_p = { (Body*)d_buf, ((Body*)d_buf) + nBodies };

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0; 

  for (int iter = 1; iter <= nIters; iter++) {
    Timer tmr;
    tmr.start();
    cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.vel, dt, nBodies);
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += p.vel[i].x*dt;
      p.pos[i].y += p.vel[i].y*dt;
      p.pos[i].z += p.vel[i].z*dt;
    }
    tmr.stop();
    const double tElapsed = tmr.getTimeMillisecond();
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
  }
  double avgTime = totalTime / (double)(nIters-1); 

#ifdef JSON_OUTPUT
  printf("{\"nbodies\":%d, \"rate\":%.3lf, \"time\": %.3lf}\n", nBodies, 1e-6 * nBodies * nBodies / avgTime, avgTime);
#else
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-6 * nBodies * nBodies / avgTime);
#endif
  free(buf);
  cudaFree(d_buf);
}
