# mpm3d

Implementing mpm3d, which was originally written in [Taichi](https://github.com/taichi-dev/taichi), using CUDA C++.

You can change configures in mpm3d.cuh (e.g., change 'dim' from 3 to 2, or
change 'Real' from float to double). It should compile correctly.

## How to build

If you don't need gui, you can delete gui.cu and gui.cuh, remove gui from main.cu,  and remove gui.cu and glfw from CMakeList.txt. 

In this case, you don't need to install glfw.

### Windows

Open this project with either Visual Studio or CLion.

See [CUDA projects in CLion
](https://www.jetbrains.com/help/clion/cuda-projects.html).

Make sure you have the newest version of MSVC, or you may get error when building Eigen.

You need to install glfw using [vcpkg](https://github.com/microsoft/vcpkg).

```
./vcpkg install glfw3:x64-windows
./vcpkg integrate install
```

### Ubuntu

Install glfw first.

```
sudo apt install libglfw3-dev
```

Then you can open this project with CLion (recommended).

See [CUDA projects in CLion
](https://www.jetbrains.com/help/clion/cuda-projects.html).

You can also build with Makefile.

```
make run
```

```
make run-release
```

```
make run-debug
```

Make sure you have installed CMake and added it to PATH, otherwise you need to 

set 'CMAKE' in the Makefile to the path to the CMake executable.
