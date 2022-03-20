cmake_minimum_required(VERSION 3.20)

project(MPM LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

add_executable(MPM2D "src/mpm2d.cu")

set_target_properties(MPM2D PROPERTIES CUDA_ARCHITECTURES 86)

target_compile_options(MPM2D PRIVATE --expt-relaxed-constexpr)

if(CMAKE_BUILD_TYPE EQUAL "Release")
    target_compile_definitions(MPM2D PUBLIC EIGEN_NO_DEBUG)
endif()

add_executable(MPM3D "src/mpm3d.cu")

set_target_properties(MPM3D PROPERTIES CUDA_ARCHITECTURES 86)

target_compile_options(MPM3D PRIVATE --expt-relaxed-constexpr)

if(CMAKE_BUILD_TYPE EQUAL "Release")
    target_compile_definitions(MPM3D PUBLIC EIGEN_NO_DEBUG)
endif()
