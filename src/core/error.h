#pragma once
// ─────────────────────────────────────────────────────────────────────
// error.h – CUDA and OptiX error-checking macros
// ─────────────────────────────────────────────────────────────────────
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <optix.h>

// ── CUDA error check ────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t rc = (call);                                           \
        if (rc != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",              \
                    cudaGetErrorString(rc), (int)rc, __FILE__, __LINE__);  \
            throw std::runtime_error(std::string("CUDA error: ") +        \
                                     cudaGetErrorString(rc));              \
        }                                                                  \
    } while (0)

#define CUDA_CHECK_NOTHROW(call)                                           \
    do {                                                                   \
        cudaError_t rc = (call);                                           \
        if (rc != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s (%d) at %s:%d\n",              \
                    cudaGetErrorString(rc), (int)rc, __FILE__, __LINE__);  \
        }                                                                  \
    } while (0)

// Check last CUDA error (async kernel launch errors).
#define CUDA_SYNC_CHECK()                                                  \
    do {                                                                   \
        cudaDeviceSynchronize();                                           \
        cudaError_t rc = cudaGetLastError();                               \
        if (rc != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA sync error %s (%d) at %s:%d\n",         \
                    cudaGetErrorString(rc), (int)rc, __FILE__, __LINE__);  \
            throw std::runtime_error(std::string("CUDA sync error: ") +   \
                                     cudaGetErrorString(rc));              \
        }                                                                  \
    } while (0)

// ── OptiX error check ───────────────────────────────────────────────

#define OPTIX_CHECK(call)                                                  \
    do {                                                                   \
        OptixResult rc = (call);                                           \
        if (rc != OPTIX_SUCCESS) {                                         \
            fprintf(stderr, "OptiX error %d at %s:%d\n",                   \
                    (int)rc, __FILE__, __LINE__);                          \
            throw std::runtime_error(std::string("OptiX error: ") +       \
                                     std::to_string((int)rc));             \
        }                                                                  \
    } while (0)

#define OPTIX_CHECK_LOG(call, log, log_size)                               \
    do {                                                                   \
        OptixResult rc = (call);                                           \
        if (rc != OPTIX_SUCCESS) {                                         \
            fprintf(stderr, "OptiX error %d at %s:%d\nLog: %s\n",         \
                    (int)rc, __FILE__, __LINE__, (log));                   \
            throw std::runtime_error(std::string("OptiX error: ") +       \
                                     std::to_string((int)rc));             \
        }                                                                  \
        if ((log_size) > 1) {                                              \
            fprintf(stderr, "OptiX log at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, (log));                             \
        }                                                                  \
    } while (0)
