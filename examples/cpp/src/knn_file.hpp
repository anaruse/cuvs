#ifndef __CAGRA_KNN_FILE_HPP__
#define __CAGRA_KNN_FILE_HPP__
#include <fstream>

namespace cagra {
namespace knn_file {
template <class T>
void save(
    const std::string file_path,
    const T* const neighbor_data_ptr,
    const std::size_t size,
    const std::size_t degree
    ) {
  std::ofstream ofs(file_path);

  ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
  ofs.write(reinterpret_cast<const char*>(&degree), sizeof(degree));

  ofs.write(
      reinterpret_cast<const char*>(neighbor_data_ptr),
      sizeof(T) * degree * size
      );
  ofs.close();
}

inline void load_knn_info(
    const std::string file_path,
    std::size_t &size,
    std::size_t &degree
    ) {
  std::ifstream ifs(file_path);
  if (!ifs) {
      throw std::runtime_error("File not exist : " + file_path + " (`" + __func__ + "` in " + __FILE__ + ")");
  }
  ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
  ifs.read(reinterpret_cast<char*>(&degree), sizeof(degree));
  ifs.close();
}

template <class T>
void load(
    const std::string file_path,
    T* const neighbor_data_ptr,
    const std::uint32_t input_knn_degree = 0
    ) {
  std::ifstream ifs(file_path);
  if (!ifs) {
    throw std::runtime_error("File not exist : " + file_path + " (`" + __func__ + "` in " + __FILE__ + ")");
  }

  std::size_t size, degree;

  ifs.read(reinterpret_cast<char*>(&size), sizeof(size));
  ifs.read(reinterpret_cast<char*>(&degree), sizeof(degree));

  std::size_t load_knn_degree = degree;
  if (degree > input_knn_degree && input_knn_degree != 0) {
    load_knn_degree = input_knn_degree;
  }

  for (std::size_t i = 0; i < size; i++) {
    ifs.read(
        reinterpret_cast<char*>(neighbor_data_ptr + i * load_knn_degree),
        sizeof(T) * load_knn_degree
        );
    if (load_knn_degree != degree) {
      ifs.seekg((degree - load_knn_degree) * sizeof(T), std::ios_base::cur);
    }
    if (i % (size / 1000) == 0) {
        std::fprintf(stderr, "# Loading kNN Graph (%3.1f %%)\r", static_cast<double>(i) / size * 100);
    }
  }
  std::fprintf(stderr, "\n");
  ifs.close();
}
} // namespace xvec_io
} // namespace cagra
#endif
