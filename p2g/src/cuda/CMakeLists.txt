cmake_minimum_required(VERSION 3.20)

project(P2G LANGUAGES CUDA CXX)

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

add_executable(P2G "src/p2g.cu")

set_target_properties(P2G PROPERTIES CUDA_ARCHITECTURES 86)

target_compile_options(P2G PRIVATE --expt-relaxed-constexpr)

if(CMAKE_BUILD_TYPE EQUAL "Release")
    target_compile_definitions(P2G PUBLIC EIGEN_NO_DEBUG)
endif()

