cmake_minimum_required(VERSION 3.28.3)

project(JPEG VERSION 0.0.1 LANGUAGES CXX)

option(USE_CUDA "Build for Nvidia GPU" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    set(CUDACXX "/usr/bin/nvcc")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/output")
set(LIBRAW_PATH "${CMAKE_SOURCE_DIR}/external/LibRaw" CACHE PATH "Path to LibRaw sources")

include(CTest)
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(tests)


