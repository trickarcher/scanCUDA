#include "awkward_offset_arrays.h"
#include <iostream>
#include <cmath>

namespace awkward {

template<typename T, typename C>
AwkwardOffsetArrayCuda<T, C>::AwkwardOffsetArrayCuda() {

}

template<typename T, typename C>
void AwkwardOffsetArrayCuda<T, C>::awkward_listarray_compact_offsets(T **tooffsets,
                                                                     C *fromstarts,
                                                                     C *fromstops,
                                                                     int64_t startsoffset,
                                                                     int64_t stopsoffset,
                                                                     int64_t length) {

  this->fromstarts = fromstarts;
  this->fromstops = fromstops;
  this->startoffset = startsoffset;
  this->stopoffset = stopsoffset;
  this->length = length;
  this->tooffsets = new T[length + 1];
  std::cout << "Assigned" << std::endl;
  SIZE_C = sizeof(C) * length;
  SIZE_T = sizeof(T) * (length + 1);

  HANDLE_ERROR(cudaMalloc((void **) &d_fromstarts, SIZE_C));
  HANDLE_ERROR(cudaMemcpy(d_fromstarts, fromstarts, SIZE_C, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void **) &d_fromstops, SIZE_C));
  HANDLE_ERROR(cudaMemcpy(d_fromstops, fromstops, SIZE_C, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void **) &d_tooffsets, SIZE_T));

  if (length > 1024) {
    blocks_per_grid = dim3(ceil(((float) length) / 1024.0), 1, 1); // length / 1024 is (N/B)
    threads_per_block = dim3(1024, 1, 1);                             // 512 is B/2
  } else {
    blocks_per_grid = dim3(1, 1, 1);
    threads_per_block = dim3(length, 1, 1);
  }

  compute_offsets_kernel_wrapper();

  HANDLE_ERROR(cudaMalloc((void **) &d_in, SIZE_T));
  HANDLE_ERROR(cudaMalloc((void **) &d_out, SIZE_T));
  HANDLE_ERROR(cudaMemcpy(d_in, d_tooffsets, SIZE_T, cudaMemcpyDeviceToDevice));

  compact_offsets_kernel_wrapper();

  (*tooffsets) = this->tooffsets;
}

template
class AwkwardOffsetArrayCuda<int64_t, int32_t>;

template
class AwkwardOffsetArrayCuda<int16_t, int8_t>;

template
class AwkwardOffsetArrayCuda<int32_t, int16_t>;

template
class AwkwardOffsetArrayCuda<int, int>;

template
class AwkwardOffsetArrayCuda<int64_t, int8_t>;
}
