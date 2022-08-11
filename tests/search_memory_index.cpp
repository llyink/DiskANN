// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <thread>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "aux_utils.h"
#include "index.h"
#include "memory_mapper.h"
#include "utils.h"
#include "insert_till_next_checkpoint.h"
#include "load_aligned_bin_part.h"
#include <time.h>
#include <timer.h>
#include <windows.h>

namespace po = boost::program_options;
using namespace std;

template<typename T>
int search_memory_index(diskann::Metric& metric,
                        diskann::Index<T, uint32_t>& index,
                        const std::string&           result_path_prefix,
                        const std::string&           query_file,
                        std::string& truthset_file, const unsigned num_threads,
                        const unsigned               recall_at,
                        const std::vector<unsigned>& Lvec, const bool dynamic,
                        const bool tags, const unsigned qps_setting) {
  // Load the query file
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);
  std::cout << "query_num is " << query_num << std::endl;  
  std::cout << "query_dim is " << query_dim << std::endl;  
  std::cout << "query_aligned_dim is " << query_aligned_dim << std::endl;  

  // Check for ground truth
  bool calc_recall_flag = false;
  if (truthset_file != std::string("null") && file_exists(truthset_file)) {
    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  } else {
    diskann::cout << " Truthset file " << truthset_file
                  << " not found. Not computing recall." << std::endl;
  }

  std::cout << "Index loaded" << std::endl;
  if (metric == diskann::FAST_L2)
    index.optimize_index_layout();

  std::cout << "Using " << num_threads << " threads to search" << std::endl;
  diskann::Parameters paras;
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  if (tags) {
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(20) << "Mean Latency (mus)" << std::setw(15)
              << "99.9 Latency";
  } else {
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Avg dist cmps" << std::setw(20)
              << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
  }
  if (calc_recall_flag)
    std::cout << std::setw(12) << recall_string;
  std::cout << std::endl;
  std::cout << "==============================================================="
               "=================="
            << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());
  std::vector<float>                 latency_stats(query_num, 0);
  std::vector<unsigned>              cmp_stats;
  if (not tags) {
    cmp_stats = std::vector<unsigned>(query_num, 0);
  }
  uint32_t* query_result_tags;
  if (tags) {
    query_result_tags = new uint32_t[recall_at * query_num];
  }

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }

    query_result_ids[test_id].resize(recall_at * query_num);
    std::vector<T*> res = std::vector<T*>();

    auto s = std::chrono::high_resolution_clock::now();
    float time = 0;
    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      if (metric == diskann::FAST_L2) {
        index.search_with_optimized_layout(
            query + i * query_aligned_dim, recall_at, L,
            query_result_ids[test_id].data() + i * recall_at);
      } else if (tags) {
        auto t1 = std::chrono::high_resolution_clock::now();  
        index.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                               query_result_tags + i * recall_at, nullptr, res);
        auto t2 = std::chrono::high_resolution_clock::now();  
        std::chrono::duration<double> diff = t2 - t1;         
        if (diff.count() * 1000000 > time) {                  
          time = diff.count() * 1000000;                      
        }
        for (int64_t r = 0; r < (int64_t) recall_at; r++) {
          query_result_ids[test_id][recall_at * i + r] =
              *(query_result_tags + recall_at * i + r);
        }
        
      } else {
        cmp_stats[i] =
            index
                .search(query + i * query_aligned_dim, recall_at, L,
                        query_result_ids[test_id].data() + i * recall_at)
                .second;
      }
      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000000;

      Sleep(1000 / qps_setting - diff.count() * 1000);  
    }
    std::cout << "Maximum search time per searchlist is " << time << "us"
              << std::endl;  
    std::chrono::duration<double> diff =
        std::chrono::high_resolution_clock::now() - s;
    float qps = (query_num / diff.count());

    float recall = 0;
    if (calc_recall_flag)
      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         query_result_ids[test_id].data(),
                                         recall_at, recall_at);

    std::sort(latency_stats.begin(), latency_stats.end());
    float mean_latency =
        std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) /
        query_num;

    float avg_cmps =
        (float) std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) /
        (float) query_num;

    if (tags) {
      std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(20)
                << (float) mean_latency << std::setw(15)
                << (float) latency_stats[(_u64) (0.999 * query_num)];
    } else {
      std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
                << avg_cmps << std::setw(20) << (float) mean_latency
                << std::setw(15)
                << (float) latency_stats[(_u64) (0.999 * query_num)];
    }
    if (calc_recall_flag)
      std::cout << std::setw(12) << recall;
    std::cout << std::endl;
  }

  std::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }
    std::string cur_result_path =
        result_path_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                            query_num, recall_at);
    test_id++;
  }

  diskann::aligned_free(query);
  if (tags)
    delete[] query_result_tags;

  return 0;
}

int main(int argc, char** argv) {
  std::string           data_type, dist_fn, result_path, query_file, gt_file;
  unsigned              num_threads, K, qps_setting;
  std::vector<unsigned> Lvec;
  bool                  dynamic, tags;
  size_t                max_points;
  float                 insert_percentage, build_percentage, delete_percentage;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2/cosine>");
    desc.add_options()("result_path",
                       po::value<std::string>(&result_path)->required(),
                       "Path prefix for saving results of the queries");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "gt_file",
        po::value<std::string>(&gt_file)->default_value(std::string("null")),
        "ground truth file for the queryset");
    desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                       "Number of neighbors to be returned");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()("num_threads,T",
                       po::value<uint32_t>(&num_threads)->default_value(1),
                       "Number of threads used for building index (defaults to "
                       "omp_get_num_procs())");
    desc.add_options()("dynamic",
                       po::value<bool>(&dynamic)->default_value(false),
                       "Whether the index is dynamic. Default false.");
    desc.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                       "Whether to search with tags. Default false.");
    desc.add_options()("qps_setting,qps",
                       po::value<uint32_t>(&qps_setting)->default_value(1),
                       "qps_setting for qps ");
    desc.add_options()(
        "max_points", po::value<uint64_t>(&max_points)->default_value(0),
        "These number of points from the file"
        "points_to_skip");
    desc.add_options()("insert_percentage",
                       po::value<float>(&insert_percentage)->required(),
                       "Batch build will be called on these set of points");
    desc.add_options()("build_percentage",
                       po::value<float>(&build_percentage)->required(),
                       "build will be called on these set of points");
    desc.add_options()("delete_percentage",
                       po::value<float>(&delete_percentage)->required(), "");

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

  diskann::Metric metric;
  if ((dist_fn == std::string("mips")) && (data_type == std::string("float"))) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else if ((dist_fn == std::string("fast_l2")) &&
             (data_type == std::string("float"))) {
    metric = diskann::Metric::FAST_L2;
  } else {
    std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                 "supported in general, and mips/fast_l2 only for floating "
                 "point data."
              << std::endl;
    return -1;
  }

  if (dynamic && not tags) {
    std::cerr
        << "Tags must be enabled while searching dynamically built indices"
        << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("float")) {
      const unsigned      C = 500;
      const bool          saturate_graph = false;
      diskann::Parameters params;
      params.Set<unsigned>("L", 25);
      params.Set<unsigned>("R", 128);
      params.Set<unsigned>("C", C);
      params.Set<float>("alpha", 1.2);
      params.Set<bool>("saturate_graph", saturate_graph);
      params.Set<unsigned>("num_rnds", 1);
      params.Set<unsigned>("num_threads", 1);
      size_t dim, aligned_dim;
      size_t num_points;
      diskann::get_bin_metadata(
          "D:\\DiskANN\\build\\data\\ann_inputs\\ann_inputs_learn.fbin",
          num_points, dim);
      aligned_dim = ROUND_UP(dim, 8);
      std::cout << "num_points = " << num_points << " "
                << "dim = " << dim << std::endl;
      std::cout << "aligned_dim = " << aligned_dim << std::endl;
      using TagT = uint32_t;
      unsigned   num_frozen = 1;
      const bool enable_tags = true;
      const bool support_eager_delete = false;
      auto       num_frozen_str = getenv("TTS_NUM_FROZEN");
      if (num_frozen_str != nullptr) {
        num_frozen = std::atoi(num_frozen_str);
        std::cout << "Overriding num_frozen to" << num_frozen << std::endl;
      }

      diskann::Index<float, TagT> index(diskann::L2, dim, max_points,
                                        true, params, params, enable_tags,
                                        support_eager_delete, false);

      const size_t last_point_threshold = max_points;
      int64_t      build_point = max_points * build_percentage;
      float*       data = nullptr;
      diskann::alloc_aligned((void**) &data,
                             build_point * aligned_dim * sizeof(float),
                             8 * sizeof(float));

      std::vector<TagT> tags(build_point);
      std::iota(tags.begin(), tags.end(), static_cast<TagT>(0));

      load_aligned_bin_part(
          "D:\\DiskANN\\build\\data\\ann_inputs\\ann_inputs_learn.fbin", data,
          (size_t) 0, build_point);
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
      std::cout << "Initial non-incremental index build time for "
                << build_point << " points took " << elapsedSeconds
                << " seconds (" << build_point / elapsedSeconds
                << " points/second)\n ";

      std::thread one(insert_till_next_checkpoint<float, TagT>, std::ref(index),
                      max_points, insert_percentage, build_percentage,
                      (size_t) num_threads, data, (size_t) 64,
                      delete_percentage, std::ref(params),
                      qps_setting);

      std::thread two(search_memory_index<float>, std::ref(metric),
                      std::ref(index), std::cref(result_path),
                      std::cref(query_file), std::ref(gt_file), num_threads, K,
                      std::cref(Lvec), dynamic, true, qps_setting);
      one.join();
      two.join();

    } else {
      std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
      return -1;
    }
  } catch (std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    diskann::cerr << "Index search failed." << std::endl;
    return -1;
  }
}
