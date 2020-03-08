#include "awkward_offset_arrays.h"
#include <iostream>
#include "assert.h"
namespace awkward {

template<typename T, typename C>
__global__ void awkward_listarray_compact_offsets_cuda(T *d_in,
                                                       T *d_out,
                                                       T *d_tooffsets,
                                                       int64_t curr_step,
                                                       int64_t total_steps,
                                                       int64_t stride,
                                                       int64_t length,
                                                       bool in_out_flag) {

  int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

  int thread_id = block_id * blockDim.x + threadIdx.x;
  T sum = 0;

  if (thread_id < length) {
    if (!in_out_flag) {
      if (thread_id < stride) {
        sum = d_out[thread_id];
        d_in[thread_id] = sum;
      } else {
        sum = d_out[thread_id] + d_out[thread_id - stride];
        d_in[thread_id] = sum;
      }
    } else {
      if (thread_id < stride) {
        sum = d_in[thread_id];
        d_out[thread_id] = sum;
      } else {
        sum = d_in[thread_id] + d_in[thread_id - stride];
        d_out[thread_id] = sum;
      }
    }

    if (curr_step == total_steps) {
      d_tooffsets[thread_id] = sum;
    }
  }
}

template<typename T, typename C>
__global__ void awkward_listarray_compute_offsets_cuda(T *d_tooffsets,
                                                       C *d_fromstarts,
                                                       C *d_fromstops,
                                                       int64_t startoffset,
                                                       int64_t stopoffset,
                                                       int64_t length) {

  auto block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  auto thread_id = block_id * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    if (d_fromstops[thread_id + stopoffset] > d_fromstarts[thread_id + startoffset])
      d_tooffsets[thread_id + 1] = d_fromstops[thread_id + stopoffset] - d_fromstarts[thread_id + startoffset];
    else {
      assert("Invalid!");
//      ~AwkwardOffsetArrayCuda();
    }
  }
}

template<typename T, typename C>
void AwkwardOffsetArrayCuda<T, C>::compute_offsets_kernel_wrapper() {
  awkward_listarray_compute_offsets_cuda<T, C><<<blocks_per_grid, threads_per_block>>>(d_tooffsets,
                                                                                       d_fromstarts,
                                                                                       d_fromstops,
                                                                                       startoffset,
                                                                                       stopoffset,
                                                                                       length);
  HANDLE_ERROR(cudaFree(d_fromstarts));
  HANDLE_ERROR(cudaFree(d_fromstops));
}

template<typename T, typename C>
void AwkwardOffsetArrayCuda<T, C>::compact_offsets_kernel_wrapper() {
  int64_t stride = 1;
  int total_steps = ceil(log2(static_cast<float>(length)));
  for (size_t curr_step = 1; curr_step <= total_steps; curr_step++) {
    bool in_out_flag = (curr_step % 2) != 0;
    awkward_listarray_compact_offsets_cuda<T, C><<<blocks_per_grid, threads_per_block>>>(d_in,
                                                                                         d_out,
                                                                                         d_tooffsets,
                                                                                         curr_step,
                                                                                         total_steps,
                                                                                         stride,
                                                                                         length,
                                                                                         in_out_flag);
    stride = stride * 2;
  }

  HANDLE_ERROR(cudaMemcpy(tooffsets, d_tooffsets, SIZE_T, cudaMemcpyDeviceToHost));
  this->tooffsets[length] += this->tooffsets[length - 1];

  HANDLE_ERROR(cudaFree(d_tooffsets));
  HANDLE_ERROR(cudaFree(d_in));
  HANDLE_ERROR(cudaFree(d_out));
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