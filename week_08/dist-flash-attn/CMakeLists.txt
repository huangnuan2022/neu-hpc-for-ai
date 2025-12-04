
cmake_minimum_required(VERSION 3.18)
project(dist_flash_attn C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)

# Try to find NCCL; fall back to -lnccl if not found.
find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

if (NOT NCCL_LIBRARY)
    message(WARNING "NCCL library not found via find_library, assuming -lnccl is available on the link path")
    set(NCCL_LIBRARY nccl)
endif()

add_executable(dist_flash_attn
    src/main.cu
    src/flash_attn.cu)

target_include_directories(dist_flash_attn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dist_flash_attn PRIVATE
    CUDA::cudart
    ${NCCL_LIBRARY})

# Reasonable default architectures, adjust for your cluster as needed.
set_target_properties(dist_flash_attn PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86")
