#include <iostream>
#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <boost/program_options.hpp>
#include <future>
#include <thread>

#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"
namespace po = boost::program_options;
using namespace std;
using std::random_shuffle;
// load_aligned_bin modified to read pieces of the file, but using ifstream
// instead of cached_ifstream.
template<typename T>
inline void load_aligned_bin_part(const std::string& bin_file, T* data,
                                  size_t offset_points, size_t points_to_read) {
  diskann::Timer timer;
  std::ifstream  reader;
  reader.exceptions(std::ios::failbit | std::ios::badbit);
  reader.open(bin_file, std::ios::binary | std::ios::ate);
  size_t actual_file_size = reader.tellg();
  reader.seekg(0, std::ios::beg);

  int npts_i32, dim_i32;
  reader.read((char*) &npts_i32, sizeof(int));
  reader.read((char*) &dim_i32, sizeof(int));
  size_t npts = (unsigned) npts_i32;
  size_t dim = (unsigned) dim_i32;

  size_t expected_actual_file_size =
      npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
  if (actual_file_size != expected_actual_file_size) {
    std::stringstream stream;
    stream << "Error. File size mismatch. Actual size is " << actual_file_size
           << " while expected size is  " << expected_actual_file_size
           << " npts = " << npts << " dim = " << dim
           << " size of <T>= " << sizeof(T) << std::endl;
    std::cout << stream.str();
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }

  if (offset_points + points_to_read > npts) {
    std::stringstream stream;
    stream << "Error. Not enough points in file. Requested " << offset_points
           << "  offset and " << points_to_read << " points, but have only "
           << npts << " points" << std::endl;
    std::cout << stream.str();
    throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                __LINE__);
  }

  reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

  const size_t rounded_dim = ROUND_UP(dim, 8);

  for (size_t i = 0; i < points_to_read; i++) {
    reader.read((char*) (data + i * rounded_dim), dim * sizeof(T));
    memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
  }
  reader.close();

  const double elapsedSeconds = timer.elapsed() / 1000000.0;
  std::cout << "Read " << points_to_read << " points using non-cached reads in "
            << elapsedSeconds << std::endl;
}

std::string get_save_filename(const std::string& save_path,
                              size_t points_to_skip, float points_deleted,
                              float  points_inserted,
                              size_t last_point_threshold) {
  std::string final_path = save_path;
  if (points_to_skip > 0) {
    final_path += "skip" + std::to_string(points_to_skip) + "-";
  }
  if (points_deleted > 0 && points_inserted > 0) {
    final_path += "insert" + std::to_string(points_inserted) + "-" + "delete" +
                  std::to_string(points_deleted);
  } else if (points_deleted == 0 && points_inserted > 0) {
    final_path += "insert" + std::to_string(points_inserted);
  } else if (points_deleted > 0 && points_inserted == 0) {
    final_path += "delete" + std::to_string(points_deleted);
  }
  return final_path;
}
