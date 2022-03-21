cmake_minimum_required(VERSION 3.20)

project(SMALLPT LANGUAGES CUDA CXX)

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

add_executable(SMALLPT "src/main.cu")

set_target_properties(SMALLPT PROPERTIES CUDA_ARCHITECTURES 86)
target_include_directories(SMALLPT
	PUBLIC
	  ${CMAKE_SOURCE_DIR}/src/core
	  ${CMAKE_SOURCE_DIR}/src/image)
