// mapped_file.h — RAII memory-mapped file reader (Windows)

#pragma once
#include <cstddef>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// -----------------------------------------------
// MappedFile — zero-copy read-only file mapping
// -----------------------------------------------
// Opens a file via CreateFileMapping/MapViewOfFile.
// Data is paged in by the OS on first access — no explicit read calls.
// Move-only RAII: automatically unmaps and closes handles on destruction.

struct MappedFile {
    MappedFile() = default;
    ~MappedFile() { close(); }

    // Non-copyable, move-only
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept
        : data_(o.data_), size_(o.size_), mapping_(o.mapping_), file_(o.file_) {
        o.data_ = nullptr; o.size_ = 0;
        o.mapping_ = nullptr; o.file_ = nullptr;
    }
    MappedFile& operator=(MappedFile&& o) noexcept {
        if (this != &o) {
            close();
            data_ = o.data_; size_ = o.size_;
            mapping_ = o.mapping_; file_ = o.file_;
            o.data_ = nullptr; o.size_ = 0;
            o.mapping_ = nullptr; o.file_ = nullptr;
        }
        return *this;
    }

    bool open(const std::string& filepath) {
#ifdef _WIN32
        // Convert to wide string for Unicode path support
        int wlen = MultiByteToWideChar(CP_UTF8, 0, filepath.c_str(), -1, nullptr, 0);
        if (wlen <= 0) return false;
        std::wstring wpath(wlen, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, filepath.c_str(), -1, wpath.data(), wlen);

        file_ = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) { file_ = nullptr; return false; }

        LARGE_INTEGER fsize;
        if (!GetFileSizeEx(file_, &fsize) || fsize.QuadPart == 0) {
            CloseHandle(file_); file_ = nullptr;
            return false;
        }
        size_ = static_cast<size_t>(fsize.QuadPart);

        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) {
            CloseHandle(file_); file_ = nullptr; size_ = 0;
            return false;
        }

        data_ = static_cast<const char*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!data_) {
            CloseHandle(mapping_); mapping_ = nullptr;
            CloseHandle(file_); file_ = nullptr; size_ = 0;
            return false;
        }
        return true;
#else
        (void)filepath;
        return false;  // Linux/macOS: not implemented
#endif
    }

    void close() {
#ifdef _WIN32
        if (data_)    { UnmapViewOfFile(data_); data_ = nullptr; }
        if (mapping_) { CloseHandle(mapping_); mapping_ = nullptr; }
        if (file_)    { CloseHandle(file_); file_ = nullptr; }
        size_ = 0;
#endif
    }

    bool        is_open() const { return data_ != nullptr; }
    const char* data()    const { return data_; }
    size_t      size()    const { return size_; }

private:
    const char* data_    = nullptr;
    size_t      size_    = 0;
#ifdef _WIN32
    HANDLE      mapping_ = nullptr;
    HANDLE      file_    = nullptr;
#endif
};
