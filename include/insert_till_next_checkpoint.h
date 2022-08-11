#include<iostream>
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
#include <windows.h>

namespace po = boost::program_options;
using namespace std;
template<typename T, typename TagT>
void insert_till_next_checkpoint(diskann::Index<T, TagT>& index,
                                 size_t max_points, float insert_percentage,
                                 float build_percentage, size_t thread_count,
                                 T* data, size_t aligned_dim,
                                 float                delete_percentage,
                                 diskann::Parameters& delete_params, 
                                 const unsigned qps_setting) {
  diskann::Timer insert_timer;
  int64_t        insert_points = max_points * insert_percentage;
  int64_t        delete_points = max_points * delete_percentage;
  int64_t        build_points = max_points * build_percentage;
  if (delete_percentage == insert_percentage) {
    std::cout << " random insertion and deletion" << std::endl;
    std::vector<int> streaming_numbers;
    for (int64_t j = 0; j < (int64_t) build_points - 1; j++) {
      streaming_numbers.push_back(j);
    }
    random_shuffle(streaming_numbers.begin(), streaming_numbers.end());

#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t i1 = 0; i1 < insert_points; i1++) {
      int64_t k1 = streaming_numbers[i1];
      index.insert_point(&data[k1 * aligned_dim], static_cast<TagT>(k1));
    }
    const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
    std::cout << "Insertion time " << elapsedSeconds << " seconds ("
              << insert_points / elapsedSeconds << " points/second overall, "
              << insert_points / elapsedSeconds / thread_count
              << " per thread)\n ";

    for (int64_t i2 = 0; i2 < delete_points; i2++) {
      int64_t k2 = streaming_numbers[i2];
      index.lazy_delete(k2);
    }
    auto report = index.consolidate_deletes(delete_params);
    std::cout << "#active points: " << report._active_points << std::endl
              << "max points: " << report._max_points << std::endl
              << "empty slots: " << report._empty_slots << std::endl
              << "deletes processed: " << report._slots_released << std::endl
              << "latest delete size: " << report._delete_set_size << std::endl
              << "rate: (" << delete_points / report._time
              << " points/second overall, "
              << delete_points / report._time /
                     delete_params.Get<unsigned>("num_threads")
              << " per thread)" << std::endl;
  } else if (insert_percentage > 0 && delete_percentage == 0) {
    std::cout << " only insert randomly" << std::endl;
    float time = 0;
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
    for (int64_t i = build_points; i < insert_points + build_points; i++) {

      auto t3 = std::chrono::high_resolution_clock::now();  
      index.insert_point(&data[i * aligned_dim], static_cast<TagT>(i));
      auto t4 = std::chrono::high_resolution_clock::now();  
      std::chrono::duration<double> diff = t4 - t3;         
      if (diff.count() * 1000000 > time) {  
        time = diff.count() * 1000000;      
      }                                     
      Sleep(1000 / qps_setting - diff.count() * 1000);  
    }
    std::cout << "Maximum insert time is " << time << "us"
              << std::endl;  

    const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
    std::cout << "Insertion time " << elapsedSeconds << " seconds ("
              << (max_points * insert_percentage) / elapsedSeconds
              << " points/second overall, "
              << (max_points * insert_percentage) / elapsedSeconds /
                     thread_count
              << " per thread)\n ";
  }
}

