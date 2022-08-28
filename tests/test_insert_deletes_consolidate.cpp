// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
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
#include "insert_till_next_checkpoint.h"
#include "load_aligned_bin_part.h"

namespace po = boost::program_options;
using namespace std;
using std::random_shuffle;

template<typename T, typename TagT>
void delete_from_beginning(diskann::Index<T, TagT>& index,
                           diskann::Parameters&     delete_params,
                           size_t max_points, float build_percentage,
                           size_t points_to_skip, float delete_percentage) {
  try {
    std::cout << std::endl
              << "Lazy deleting points "
              << " randomly ";
    std::vector<int> delete_numbers;
    for (int64_t j = 0; j < (int64_t) max_points; j++) {
      delete_numbers.push_back(j);
    }
    random_shuffle(delete_numbers.begin(), delete_numbers.end());
    int64_t a = 10000;
    int64_t k = 0;
    for (size_t i = points_to_skip;
         i < (points_to_skip + max_points * delete_percentage); i++) {
      int64_t m = delete_numbers[i];
      index.lazy_delete(m);
      /*if (i == (a-1)+a*k) {
        k++;
        std::cout << k << " times"
                  << "consolidate_deletes" << std::endl;
        auto report = index.consolidate_deletes(delete_params);
        std::cout << "#active points: " << report._active_points << std::endl
                  << "max points: " << report._max_points << std::endl
                  << "empty slots: " << report._empty_slots << std::endl
                  << "deletes processed: " << report._slots_released
                  << std::endl
                  << "latest delete size: " << report._delete_set_size
                  << std::endl
                  << "rate: ("
                  << (max_points * delete_percentage) / report._time
                  << " points/second overall, "
                  << (max_points * delete_percentage) / report._time /
                           delete_params.Get<unsigned>("num_threads")
                  << " per thread)" << std::endl;
      }*/
    }
    std::cout << 1 << " times"
              << "consolidate_deletes" << std::endl;
    auto report = index.consolidate_deletes(delete_params);
    std::cout << "#active points: " << report._active_points << std::endl
              << "max points: " << report._max_points << std::endl
              << "empty slots: " << report._empty_slots << std::endl
              << "deletes processed: " << report._slots_released << std::endl
              << "latest delete size: " << report._delete_set_size << std::endl
              << "rate: (" << (max_points * delete_percentage) / report._time
              << " points/second overall, "
              << (max_points * delete_percentage) / report._time /
                     delete_params.Get<unsigned>("num_threads")
              << " per thread)" << std::endl;
  } catch (std::system_error& e) {
    std::cout << "Exception caught in deletion thread: " << e.what()
              << std::endl;
  }
}

template<typename T>
void build_incremental_index(
    const std::string& data_path, const unsigned L, const unsigned R,
    const float alpha, const unsigned thread_count, size_t points_to_skip,
    size_t max_points, float insert_percentage, float build_percentage,
    const std::string& save_path, float delete_percentage,
    size_t start_deletes_after, bool concurrent, const unsigned qps_setting) {
  const unsigned      C = 500;
  const bool          saturate_graph = false;
  diskann::Parameters params;
  params.Set<unsigned>("L", L);
  params.Set<unsigned>("R", R);
  params.Set<unsigned>("C", C);
  params.Set<float>("alpha", alpha);
  params.Set<bool>("saturate_graph", saturate_graph);
  params.Set<unsigned>("num_rnds", 1);
  params.Set<unsigned>("num_threads", thread_count);
  size_t dim, aligned_dim;
  size_t num_points;

  diskann::get_bin_metadata(data_path, num_points, dim);
  aligned_dim = ROUND_UP(dim, 8);
  std::cout << "num_points = " << num_points << " "
            << "dim = " << dim << std::endl;
  std::cout << "aligned_dim = " << aligned_dim << std::endl;

  if (points_to_skip > num_points) {
    throw diskann::ANNException("Asked to skip more points than in data file",
                                -1, __FUNCSIG__, __FILE__, __LINE__);
  }
  if (max_points == 0) {
    max_points = num_points;
  }
  if (points_to_skip + max_points > num_points) {
    max_points = num_points - points_to_skip;
    std::cerr << "WARNING: Reducing max_points to " << max_points
              << " points since the data file has only that many" << std::endl;
  }

  using TagT = uint32_t;
  unsigned   num_frozen = 1;
  const bool enable_tags = true;
  const bool support_eager_delete = false;

  auto num_frozen_str = getenv("TTS_NUM_FROZEN");

  if (num_frozen_str != nullptr) {
    num_frozen = std::atoi(num_frozen_str);
    std::cout << "Overriding num_frozen to" << num_frozen << std::endl;
  }

  diskann::Index<T, TagT> index(diskann::L2, dim, max_points, true, params,
                                params, enable_tags, support_eager_delete,
                                concurrent);

  const size_t last_point_threshold = points_to_skip + max_points;

  if (max_points * insert_percentage > max_points) {
    insert_percentage = 1;
    std::cerr << "WARNING: Reducing insert index size to " << max_points
              << " points since the data file has only that many" << std::endl;
  }
  int64_t build_point = max_points * build_percentage;
  if (build_point == max_points) {
    build_point = max_points * build_percentage - 1;
  } else {
    build_point = max_points * build_percentage;
  }

  T* data = nullptr;
  diskann::alloc_aligned((void**) &data, build_point * aligned_dim * sizeof(T),
                         8 * sizeof(T));

  std::vector<TagT> tags(build_point);
  std::iota(tags.begin(), tags.end(), static_cast<TagT>(points_to_skip));

  load_aligned_bin_part(data_path, data, points_to_skip, build_point);
  std::cout << "load aligned bin succeeded" << std::endl;
  diskann::Timer timer;

  if (build_point > 0) {
    index.build(data, build_point, params, tags);
    index.enable_delete();
  } else {
    index.build_with_zero_points();
    index.enable_delete();
  }

  const double elapsedSeconds = timer.elapsed() / 1000000.0;
  std::cout << "Initial non-incremental index build time for " << build_point
            << " points took " << elapsedSeconds << " seconds ("
            << build_point / elapsedSeconds << " points/second)\n ";

  int64_t insert_index_size = max_points * insert_percentage;
  if (insert_percentage > 0) {
    if (delete_percentage > 0) {
      load_aligned_bin_part(data_path, data, points_to_skip, build_point);
    } else {
      diskann::alloc_aligned(
          (void**) &data,
          (build_point + insert_index_size) * aligned_dim * sizeof(T),
          8 * sizeof(T));
      std::vector<TagT> tags(build_point + max_points * insert_percentage);
      std::iota(tags.begin(), tags.end(), static_cast<TagT>(points_to_skip));
      load_aligned_bin_part(data_path, data, points_to_skip,
                            (build_point + insert_index_size));
    }
    insert_till_next_checkpoint(
        index, max_points, insert_percentage, build_percentage, thread_count,
        data, aligned_dim, delete_percentage, params, qps_setting);
  } else if (insert_percentage == 0 && delete_percentage > 0) {
    delete_from_beginning(index, params, max_points, build_percentage,
                          points_to_skip, delete_percentage);
  }
  const auto save_path_inc = get_save_filename(
      save_path + ".after-", points_to_skip, delete_percentage,
      insert_percentage, last_point_threshold);
  index.save(save_path_inc.c_str(), true);
  diskann::aligned_free(data);
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, data_path, index_path_prefix;
  unsigned    num_threads, R, L, qps_setting;
  float       alpha, insert_percentage, build_percentage, delete_percentage;
  size_t      points_to_skip, max_points, start_deletes_after;
  bool        concurrent;
  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips>");
    desc.add_options()("data_path",
                       po::value<std::string>(&data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
                       "Path prefix for saving index file components");
    desc.add_options()("max_degree,R",
                       po::value<uint32_t>(&R)->default_value(64),
                       "Maximum graph degree");
    desc.add_options()(
        "Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
        "Build complexity, higher value results in better graphs");
    desc.add_options()(
        "alpha", po::value<float>(&alpha)->default_value(1.2f),
        "alpha controls density and diameter of graph, set 1 for sparse graph, "
        "1.2 or 1.4 for denser graphs with lower diameter");
    desc.add_options()("num_threads,T",
                       po::value<uint32_t>(&num_threads)->default_value(1),
                       "Number of threads used for building index (defaults to "
                       "omp_get_num_procs())");
    desc.add_options()("points_to_skip",
                       po::value<uint64_t>(&points_to_skip)->required(),
                       "Skip these first set of points from file");
    desc.add_options()(
        "max_points", po::value<uint64_t>(&max_points)->default_value(0),
        "These number of points from the file are inserted after "
        "points_to_skip");
    desc.add_options()("insert_percentage",
                       po::value<float>(&insert_percentage)->required(),
                       "Batch build will be called on these set of points");
    desc.add_options()("build_percentage",
                       po::value<float>(&build_percentage)->required(),
                       "build will be called on these set of points");
    desc.add_options()("delete_percentage",
                       po::value<float>(&delete_percentage)->required(), "");
    desc.add_options()("do_concurrent",
                       po::value<bool>(&concurrent)->default_value(false), "");
    desc.add_options()(
        "start_deletes_after",
        po::value<uint64_t>(&start_deletes_after)->default_value(0), "");
    desc.add_options()("qps_setting,qps",
                       po::value<uint32_t>(&qps_setting)->default_value(1),
                       "qps_setting for qps ");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }
  try {
    if (data_type == std::string("int8"))
      build_incremental_index<int8_t>(
          data_path, L, R, alpha, num_threads, points_to_skip, max_points,
          insert_percentage, build_percentage, index_path_prefix,
          delete_percentage, start_deletes_after, concurrent, qps_setting);
    else if (data_type == std::string("uint8"))
      build_incremental_index<uint8_t>(
          data_path, L, R, alpha, num_threads, points_to_skip, max_points,
          insert_percentage, build_percentage, index_path_prefix,
          delete_percentage, start_deletes_after, concurrent, qps_setting);
    else if (data_type == std::string("float"))
      build_incremental_index<float>(
          data_path, L, R, alpha, num_threads, points_to_skip, max_points,
          insert_percentage, build_percentage, index_path_prefix,
          delete_percentage, start_deletes_after, concurrent, qps_setting);
    else
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "Caught unknown exception" << std::endl;
    exit(-1);
  }
  return 0;
}
