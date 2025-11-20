#src/CMakeLists.txt
file(GLOB source_files "*.cpp")

if(USE_CUDA)
    add_subdirectory(cuda_backend)
    add_library(backend ${source_files} ${cuda_sources})
    target_compile_definitions(backend PRIVATE USE_CUDA=1)
else()
    add_library(backend ${source_files})
    target_compile_definitions(backend PRIVATE USE_CUDA=0)

endif()