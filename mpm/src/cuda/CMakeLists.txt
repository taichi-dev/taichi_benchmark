cmake_minimum_required(VERSION 3.20)

project(MPM LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB MPM_SRC
     "src/mpm.cu"
)

add_executable(MPM ${MPM_SRC})

set_target_properties(MPM PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86)

target_compile_options(MPM PRIVATE --expt-relaxed-constexpr)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

if(CMAKE_BUILD_TYPE EQUAL "Release")
    target_compile_definitions(MPM PUBLIC EIGEN_NO_DEBUG)
endif()
