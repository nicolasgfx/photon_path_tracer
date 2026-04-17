#pragma once
// ─────────────────────────────────────────────────────────────────────
// device_buffer.h – RAII wrapper for CUDA device memory
//
// Manages cudaMalloc/cudaFree lifetime and provides typed upload/
// download.  ensure_alloc() avoids redundant re-allocation when the
// buffer is already large enough (amortization pattern from §20.1).
// ─────────────────────────────────────────────────────────────────────
#include "error.h"
#include <cstring>
#include <vector>

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { free(); }

    // Non-copyable, movable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), bytes_(other.bytes_) {
        other.ptr_   = nullptr;
        other.count_ = 0;
        other.bytes_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_   = other.ptr_;
            count_ = other.count_;
            bytes_ = other.bytes_;
            other.ptr_   = nullptr;
            other.count_ = 0;
            other.bytes_ = 0;
        }
        return *this;
    }

    // Allocate exactly n elements (frees previous allocation).
    void alloc(size_t n) {
        free();
        if (n == 0) return;
        count_ = n;
        bytes_ = n * sizeof(T);
        CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
    }

    // Allocate at least n elements.  No-op if already large enough.
    // This is the hot-path amortization pattern.
    void ensure_alloc(size_t n) {
        size_t needed = n * sizeof(T);
        if (bytes_ >= needed) {
            count_ = n;  // update logical count
            return;
        }
        alloc(n);
    }

    void free() {
        if (ptr_) {
            CUDA_CHECK_NOTHROW(cudaFree(ptr_));
            ptr_   = nullptr;
            count_ = 0;
            bytes_ = 0;
        }
    }

    // Upload from host to device.
    void upload(const T* host_data, size_t n) {
        ensure_alloc(n);
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void upload(const std::vector<T>& host_vec) {
        upload(host_vec.data(), host_vec.size());
    }

    // Download from device to host.
    void download(T* host_data, size_t n) const {
        size_t copy_n = (n < count_) ? n : count_;
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, copy_n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void download(std::vector<T>& host_vec) const {
        host_vec.resize(count_);
        download(host_vec.data(), count_);
    }

    // Zero the device memory.
    void zero() {
        if (ptr_ && bytes_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, bytes_));
        }
    }

    // Accessors.
    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }
    size_t   size() const { return count_; }
    size_t   bytes() const { return bytes_; }
    bool     empty() const { return ptr_ == nullptr; }

    // Implicit conversion to raw pointer (for kernel launch params).
    operator T*()             { return ptr_; }
    operator const T*() const { return ptr_; }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;
    size_t bytes_ = 0;
};
