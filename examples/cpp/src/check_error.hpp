#ifndef __CAGURA_CHECK_ERROR__
#define __CAGURA_CHECK_ERROR__
#include <string>
#include <stdexcept>
#include <cuda.h>

#define CAGRA_CHECK_ERROR(status) cagra_check_error((status), __FILE__, __LINE__, #status)

inline void cagra_check_error (
    const cudaError_t status,
    const char* const file_name,
    const unsigned line,
    const std::string func_name
    ) {
  if (status != cudaSuccess) {
    std::string message = "[Error] ";

    message += cudaGetErrorString(status);
    message += " (";
    message += file_name;
    message += " at l.";
    message += std::to_string(line);
    message += ", ";
    message += "\"" + func_name.substr(0, std::min(func_name.length(), 20lu)) + "...\"";
    message += ")";
    throw std::runtime_error(message);
  }
}

#endif
