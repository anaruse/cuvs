#ifndef __CAGRA_UTILS_HPP__
#define __CAGRA_UTILS_HPP__
#include <cstdint>
#include <type_traits>
#include <cfloat>
#include <cuda.h>
#include <cuda_fp16.h>

#ifndef CAGRA_HOST_DEVICE
#define CAGRA_HOST_DEVICE __host__ __device__
#endif
#ifndef CAGRA_DEVICE
#define CAGRA_DEVICE __device__
#endif

namespace cagra {
namespace utils {

template <class DATA_T> inline cudaDataType_t get_cuda_data_type();
template <> inline cudaDataType_t get_cuda_data_type<float>() {return CUDA_R_32F;}
template <> inline cudaDataType_t get_cuda_data_type<half>() {return CUDA_R_16F;}
template <> inline cudaDataType_t get_cuda_data_type<int8_t>() {return CUDA_R_8I;}
template <> inline cudaDataType_t get_cuda_data_type<uint8_t>() {return CUDA_R_8U;}
template <> inline cudaDataType_t get_cuda_data_type<uint32_t>() {return CUDA_R_32U;}
template <> inline cudaDataType_t get_cuda_data_type<uint64_t>() {return CUDA_R_64U;}

template <class T> constexpr unsigned size_of();
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<std::int8_t >() {return 1;}
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<std::uint8_t >() {return 1;}
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<std::uint16_t>() {return 2;}
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<std::uint32_t>() {return 4;}
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<std::uint64_t>() {return 8;}
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<uint4        >() {return 16;}
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<ulonglong4   >() {return 32;}
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<float        >() {return 4;}
template<> CAGRA_HOST_DEVICE constexpr unsigned size_of<half         >() {return 2;}

// max values for data types
template <class BS_T, class FP_T> union fp_conv{BS_T bs; FP_T fp;};
template <class T> CAGRA_HOST_DEVICE inline T get_max_value();
template <> CAGRA_HOST_DEVICE inline float get_max_value<float>() {return FLT_MAX;};
template <> CAGRA_HOST_DEVICE inline half  get_max_value<half>()  {return fp_conv<std::uint16_t, half>{.bs = 0x7aff}.fp;};
template <> CAGRA_HOST_DEVICE inline std::uint32_t get_max_value<std::uint32_t>()  {return 0xffffffffu;};

template <int A, int B, class=void>
struct constexpr_max {static const int value = A;};

template <int A, int B>
struct constexpr_max<A, B, std::enable_if_t<(B > A), bool>> {static const int value = B;};

} // namespace utils
} // namespace cagra

#endif
