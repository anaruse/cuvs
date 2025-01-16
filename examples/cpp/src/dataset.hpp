#ifndef __CAGRA_DATASET_HPP__
#define __CAGRA_DATASET_HPP__
#include <cstdint>
#include <fstream>
#include <string>
#include <cassert>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include "memory.hpp"
#include "utils.hpp"

namespace cagra {
namespace dataset {

enum file_format_t {
    XVECS,
    BIGANN,
    AUTO_DETECT
};

template <class DATA_T>
struct descriptor_t {
    std::string file_path;
    std::size_t dim;
    std::size_t size;

    uint32_t header_size_in_byte;
    DATA_T* data_ptr;
    cudaDataType_t dtype;
    file_format_t file_format;

    std::size_t array_size;
};

template <class DATA_T>
int get_info(
    descriptor_t<DATA_T>& desc,
    const std::string file_path,
    file_format_t file_format = AUTO_DETECT
    )
{
    desc.dtype = cagra::utils::get_cuda_data_type<DATA_T>();

    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("File not exist : " + file_path + " (`" + __func__ + "` in " + __FILE__ + ")");
    }

    // Calculate file size
    ifs.seekg(0, std::ios::end);
    const std::size_t file_size_in_byte = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    uint32_t tmp_val[2];
    ifs.read((char*)tmp_val, sizeof(uint32_t) * 2);
    // fprintf(stderr, "# tmp_val[0] = %u, tmp_val[1] = %u\n", tmp_val[0], tmp_val[1]);
    ifs.close();

    // Detect the file format
    desc.file_format = file_format;
    if (desc.file_format == AUTO_DETECT) {
        if (sizeof(uint32_t) * 2 + sizeof(DATA_T) * tmp_val[0] * tmp_val[1] == file_size_in_byte) {
            desc.file_format = BIGANN;
        } else {
            desc.file_format = XVECS;
        }
    }
    if (desc.file_format == BIGANN) {
        fprintf(stderr, "# BIGANN type file (%s)\n", file_path.c_str());
        // |--- 4 byte ---|--- 4 byte ---|--- data --- ...
        // | dataset size | dataset dim  | data 0 | data 1 | ...
        desc.size = tmp_val[0];
        desc.dim = tmp_val[1];
    } else {
        fprintf(stderr, "# Xvec type file (%s)\n", file_path.c_str());
        // |--- 4 byte ---|--- data ---
        // | dataset dim  | (index+data) 0 | (index+data) 1 | ...
        desc.dim = tmp_val[0];
        desc.size = (file_size_in_byte - 4) / desc.dim / sizeof(DATA_T) - 1;
    }

    return 0;
}

template <class DATA_T>
int load (
    descriptor_t<DATA_T>& desc,
    const std::string file_path,
    file_format_t file_format = AUTO_DETECT,
    std::string location = "normal-host"
    )
{
    get_info(desc, file_path, file_format);
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("File not exist : " + file_path + " (`" + __func__ + "` in " + __FILE__ + ")");
    }

    desc.array_size = sizeof(DATA_T) * desc.dim * desc.size;
    desc.data_ptr = (DATA_T*) cagra::memory::alloc( desc.array_size, location );

    if (desc.file_format == BIGANN) {
        // |--- 4 byte ---|--- 4 byte ---|--- data --- ...
        // | dataset size | dataset dim  | data 0 | data 1 | ...
        ifs.seekg(sizeof(uint32_t) * 2, std::ios::beg);
        ifs.read((char*)desc.data_ptr, sizeof(DATA_T) * desc.dim * desc.size + sizeof(std::uint32_t) * 2);
    } else {
        // |--- 4 byte ---|--- data ---
        // | dataset dim  | (index+data) 0 | (index+data) 1 | ...
        ifs.seekg(sizeof(uint32_t), std::ios::beg);
        for (std::size_t i = 0; i < desc.size; i++) {
            // Skip index section
            ifs.seekg(sizeof(uint32_t), std::ios::cur);
            // Load data section
            ifs.read(reinterpret_cast<char*>(desc.data_ptr + i * desc.dim), sizeof(DATA_T) * desc.dim);
        }
    }
    ifs.close();

    return 0;
}

template <class DATA_T>
void destroy(
    descriptor_t<DATA_T>& desc
    )
{
    cagra::memory::free( desc.data_ptr );
}

} // namespace dataset
} // namespace cagra
#endif
