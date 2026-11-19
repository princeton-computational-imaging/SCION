// Common CUDA constants + utilities used across kernels.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace scion_cuda {

constexpr int BLOCK_SIZE_ASSEMBLE = 256;

// Generic ceil-div for grid-size math.
inline __host__ __device__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

}  // namespace scion_cuda
