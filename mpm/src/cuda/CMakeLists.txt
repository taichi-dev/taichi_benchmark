cmake_minimum_required(VERSION 3.12)

project(MPM3D LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB MPM_SRC
     "src/main.cu"
)

add_executable(MPM3D ${MPM_SRC})

set_target_properties(MPM3D PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86)

target_compile_options(MPM3D PRIVATE --expt-relaxed-constexpr)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

if(CMAKE_BUILD_TYPE EQUAL "Release")
    target_compile_definitions(MPM3D PUBLIC EIGEN_NO_DEBUG)
endif()
