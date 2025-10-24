cmake_minimum_required(VERSION 3.28.3)

project(JPEG VERSION 0.0.1 LANGUAGES CXX)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/output")


include(CTest)
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(tests)


