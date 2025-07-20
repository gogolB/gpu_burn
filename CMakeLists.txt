cmake_minimum_required(VERSION 3.20)
project(GpuBurn LANGUAGES CXX CUDA)

# Find CUDA using the modern CUDAToolkit module
find_package(CUDAToolkit 12.9 REQUIRED)

# Set CUDA architectures for different GPU generations
# SM 7.0 = Volta (V100)
# SM 7.5 = Turing (RTX 20xx, T4)
# SM 8.0 = Ampere (A100)
# SM 8.6 = Ampere (RTX 30xx, A40)
# SM 8.9 = Ada Lovelace (RTX 40xx, L40)
# SM 9.0 = Hopper (H100)
# SM 10.0 = Blackwell (H200, L200)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 100)

# Standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo -Wno-deprecated-gpu-targets")

# Enable separate compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(include)

# Collect source files
file(GLOB HOST_SOURCES "src/host/*.cpp")
file(GLOB KERNEL_SOURCES "src/kernels/*.cu" "src/kernels/*.cpp")
file(GLOB VALIDATION_SOURCES "src/validation/*.cpp" "src/validation/*.cu")
file(GLOB MONITORING_SOURCES "src/monitoring/*.cpp" "src/monitoring/*.cu")

# Add the main executable
add_executable(gpu_burn 
    ${HOST_SOURCES}
    ${KERNEL_SOURCES}
    ${VALIDATION_SOURCES}
    ${MONITORING_SOURCES}
)

# Link CUDA libraries
target_link_libraries(gpu_burn PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::cusolver
    nvidia-ml  # NVML library for GPU monitoring
)

# Add test executable (optional)
option(BUILD_TESTS "Build test suite" ON)
if(BUILD_TESTS)
    file(GLOB TEST_SOURCES "tests/*.cpp" "tests/*.cu")
    if(TEST_SOURCES)
        add_executable(gpu_burn_tests ${TEST_SOURCES})
        target_link_libraries(gpu_burn_tests PRIVATE 
            CUDA::cudart
            CUDA::cublas
            CUDA::curand
        )
    endif()
endif()

# Installation rules
install(TARGETS gpu_burn DESTINATION bin)