
From nVidia official blog https://developer.nvidia.com/blog/six-ways-saxpy/

Compilation
```
nvcc -O3 thrust.cu -o saxpy_thrust
nvcc -O3 cublas.cu -o saxpy_cublas -lcublas
```
