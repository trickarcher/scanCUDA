#include "awkward_offset_arrays.h"
#include <iostream>
#include <cmath>
#include <limits>

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
  SIZE_C = sizeof(C) * length;
  SIZE_T = sizeof(T) * (length + 1);

  HANDLE_ERROR(cudaMalloc((void **) &d_fromstarts, SIZE_C));
  HANDLE_ERROR(cudaMemcpy(d_fromstarts, fromstarts, SIZE_C, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void **) &d_fromstops, SIZE_C));
  HANDLE_ERROR(cudaMemcpy(d_fromstops, fromstops, SIZE_C, cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void **) &d_tooffsets, SIZE_T));

  if (length > 1024) {
    // A Ceil function enables us to have the scan in power of 2 while keeping overhead to a minimum
    // A better approach would be to call the scan for length / 1024 blocks and then compute the rest via CPU
    // which is guaranteed to be faster for remainder_length < 1024
    blocks_per_grid = dim3(ceil(((float) length) / 1024.0), 1, 1);
    threads_per_block = dim3(1024, 1, 1);
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

template<typename T, typename C>
bool AwkwardOffsetArrayCuda<T, C>::check_correctness(int64_t length, int64_t startoffset, int64_t stopoffset) {

  T *tooffsets = new T[length + 1];
  C *fromstarts = new C[length];
  C *fromstops = new C[length];

  srand(time(0));

  // Generate some rough data
  C max_limit = std::numeric_limits<C>::max();
  for (int64_t i = 0; i < length; i++) {
    C val_1 = rand() % max_limit;
    C val_2 = rand() % max_limit;
    fromstarts[i] = (val_1 <= val_2) ? val_1 : val_2;
    fromstops[i] = (val_1 >= val_2) ? val_1 : val_2;
  }

  awkward::AwkwardOffsetArrayCuda<T, C> arr;
  arr.awkward_listarray_compact_offsets(&tooffsets, fromstarts, fromstops, startoffset, stopoffset, length);

  int *cpu_tooffsets = new int[length + 1];
  cpu_tooffsets[0] = 0;
  for (int i = 0; i < length; i++) {
    C stop, start;
    if(stopoffset + i < length  && startoffset + i < length) {
      stop = fromstops[stopoffset + i];
      start = fromstarts[startoffset + i];
    }
    else {
      stop = 0;
      start = 0;
    }
    if(stop < start)
      return false;

    cpu_tooffsets[i + 1] = cpu_tooffsets[i] + (stop - start);
  }
  bool flag = true;
  for (int i = 0; i <= length; i++) {
    if (cpu_tooffsets[i] != tooffsets[i]) {
      flag = false;
    }
  }

  delete(tooffsets);
  delete(fromstarts);
  delete(fromstops);
  return flag;
}


template
class AwkwardOffsetArrayCuda<int64_t, int32_t>;

template
class AwkwardOffsetArrayCuda<int64_t, int8_t>;

template
class AwkwardOffsetArrayCuda<int64_t, int16_t>;

}
