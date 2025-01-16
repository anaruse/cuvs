#ifndef __CAGRA_MEMORY_HPP__
#define __CAGRA_MEMORY_HPP__

#include <iostream>
#include <utility>
#include <vector>
#include <tuple>
#include <cassert>
#include <algorithm>

#include <sys/mman.h>

#include "check_error.hpp"

namespace cagra {
namespace memory {

constexpr int debug = 1;

enum mtype_t {
    NONE,           // *** INVALID ***
    NORMAL_HOST,    // malloc()
    HUGEPAGE_HOST,  // mmap() + madvice()
    PINNED_HOST,    // cudaMallocHost()
    MANAGED_HOST,   // cudaMallocManaged()
    MANAGED,        // cudaMallocManaged()
    DEVICE,         // cudaMalloc()
};

std::string mtypeToString(mtype_t mtype)
{
    switch (mtype) {
    case mtype_t::NONE:          return "NONE (INVALID)";
    case mtype_t::NORMAL_HOST:   return "NORMAL_HOST";
    case mtype_t::HUGEPAGE_HOST: return "HUGEPAGE_HOST";
    case mtype_t::PINNED_HOST:   return "PINNED_HOST";
    case mtype_t::MANAGED_HOST:  return "MANAGED_HOST";
    case mtype_t::MANAGED:       return "MANAGED";
    case mtype_t::DEVICE:        return "DEVICE";
    default:                     return "Unknown";
    }
}

using PtrMtypeSize = std::tuple<void*, mtype_t, size_t>;

std::vector<PtrMtypeSize> ptr_mtype_size_list;

void* alloc(size_t size, mtype_t mtype)
{
    void *ptr = nullptr;
    switch (mtype) {
    case mtype_t::NORMAL_HOST:
        ptr = ::malloc(size);
        break;
    case mtype_t::HUGEPAGE_HOST:
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert( ptr != MAP_FAILED );
        assert( madvise(ptr, size, MADV_HUGEPAGE) == 0 );
        break;
    case mtype_t::PINNED_HOST:
        CAGRA_CHECK_ERROR(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
        break;
    case mtype_t::MANAGED_HOST:
        CAGRA_CHECK_ERROR(cudaMallocManaged(&ptr, size));
        // Pin down location to HOST memory
        CAGRA_CHECK_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        // Load PTE to GPU 0?
        CAGRA_CHECK_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, /*device=*/ 0));
        break;
    case mtype_t::MANAGED:
        CAGRA_CHECK_ERROR(cudaMallocManaged(&ptr, size));
        break;
    case mtype_t::DEVICE:
        CAGRA_CHECK_ERROR(cudaMalloc(&ptr, size));
        break;
    default:
        fprintf(stderr, "*** ERROR (%s, %s, %d) ***, Unknown mtype (%d)\n", __FILE__, __func__, __LINE__,
                mtype );
        exit(-1);
        break;
    }
    if ( debug ) {
        fprintf(stderr, "# stats (%s, %s, %d): ptr:%p, mtype:%s, size:%lu\n", __FILE__, __func__, __LINE__,
                ptr, mtypeToString(mtype).c_str(), size );
    }

    ptr_mtype_size_list.push_back( std::make_tuple( ptr, mtype, size ) );
    return ptr;
}

void* alloc(size_t size, std::string location = "normal-host")
{
    auto mtype = mtype_t::NONE;
    if ( location == "normal-host" ) {
        mtype = mtype_t::NORMAL_HOST;
    } else if ( location == "hugepage-host" ) {
        mtype = mtype_t::HUGEPAGE_HOST;
    } else if ( location == "pinned-host" ) {
        mtype = mtype_t::PINNED_HOST;
    } else if ( location == "managed-host" ) {
        mtype = mtype_t::MANAGED_HOST;
    } else if ( location == "managed" ) {
        mtype = mtype_t::MANAGED;
    } else if ( location == "device" ) {
        mtype = mtype_t::DEVICE;
    } else {
        fprintf(stderr, "*** ERROR (%s, %s, %d) ***, Unknown location (%s)\n", __FILE__, __func__, __LINE__,
                location.c_str() );
        exit(-1);
    }
    return alloc(size, mtype);
}

void free(void *ptr)
{
    auto it = std::find_if(ptr_mtype_size_list.begin(), ptr_mtype_size_list.end(),
                           [ptr=ptr](PtrMtypeSize& a) { return std::get<0>(a) == ptr; });
    if (it == ptr_mtype_size_list.end()) {
        fprintf(stderr, "*** ERROR (%s, %s, %d) ***, Unknown ptr (%p)\n", __FILE__, __func__, __LINE__,
                ptr );
        exit(-1);
    }
    auto a = (PtrMtypeSize)(*it);
    assert( ptr == std::get<0>(a) );
    auto mtype = std::get<1>(a);
    auto size = std::get<2>(a);
    if ( debug ) {
        fprintf(stderr, "# stats (%s, %s, %d): ptr:%p, mtype:%s, size:%lu\n", __FILE__, __func__, __LINE__,
                ptr, mtypeToString(mtype).c_str(), size );
    }

    if ( mtype == mtype_t::NORMAL_HOST ) {
        ::free( ptr );
    } else if ( mtype == mtype_t::HUGEPAGE_HOST ) {
        assert( munmap(ptr, size) == 0);
    } else if ( mtype == mtype_t::PINNED_HOST ) {
        CAGRA_CHECK_ERROR( cudaFreeHost(ptr) );
    } else if ( mtype == mtype_t::MANAGED_HOST ) {
        CAGRA_CHECK_ERROR( cudaMemAdvise(ptr, size, cudaMemAdviseUnsetAccessedBy, /*device=*/ 0) );
        CAGRA_CHECK_ERROR( cudaMemAdvise(ptr, size, cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId) );
        CAGRA_CHECK_ERROR( cudaFree(ptr) );
    } else if ( mtype == mtype_t::MANAGED ) {
        CAGRA_CHECK_ERROR( cudaFree(ptr) );
    } else if ( mtype == mtype_t::DEVICE ) {
        CAGRA_CHECK_ERROR( cudaFree(ptr) );
    } else {
        fprintf(stderr, "*** ERROR (%s, %s, %d) ***, Unknown mtype (%d)\n", __FILE__, __func__, __LINE__,
                mtype );
        exit(-1);
    }
    ptr_mtype_size_list.erase(it);
}

} // namespace memory
} // namespace cagra
#endif
