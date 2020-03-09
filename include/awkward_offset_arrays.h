#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <random>

#pragma once

/**
 * This is a utility funstion for checking CUDA Errors,
 * NOTE: This function was taken from a blog www.beechwood.eu
 * @param err
 * @param file
 * @param line
 */
static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
  if (err != cudaSuccess) {
    int aa = 0;
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    scanf("%d", &aa);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

namespace awkward {

template<typename T, typename C>
class AwkwardOffsetArrayCuda {
 private:

  int64_t length;

  C *fromstarts;
  C *fromstops;
  T *tooffsets;

  // Anything starting with a d_ prefix is destined to run on the GPU so these are the device arrays
  C *d_fromstarts;
  C *d_fromstops;
  T *d_tooffsets;
  T *d_in;
  T *d_out;

  int64_t startoffset;
  int64_t stopoffset;

  // Store the Size in bytes of C and T data types
  int64_t SIZE_C;
  int64_t SIZE_T;

  // These are the Kernel launch parameters, currently launching 1024 threads per block if length is > 1024
  dim3 threads_per_block;
  dim3 blocks_per_grid;

 public:
  /**
   * @brief Default Constructor, does nothing!
   */
  AwkwardOffsetArrayCuda<T, C>();

  /**
   * @brief This function performs the Hillis Steele Scan, calls two kernel wrappers which in turn
   * calls the device kernels which perform the scan operation
   * @param tooffsets
   * @param fromstarts
   * @param fromstops
   * @param startsoffset
   * @param stopsoffset
   * @param length
   */
  void awkward_listarray_compact_offsets(T **tooffsets,
                                         C *fromstarts,
                                         C *fromstops,
                                         int64_t startsoffset,
                                         int64_t stopsoffset,
                                         int64_t length);

  /**
   * @brief This is a kernel wrapper for awkward_listarray_compute_offsets_cuda
   * Calls the Kernel and frees d_fromstarts and d_fromstops which is no longer required
   */
  void compute_offsets_kernel_wrapper();

  /**
   * @brief This is a kernel wrapper for awkward_listarray_compact_offsets_cuda
   * Calls the Kernel and free d_in, d_out, and d_offsets after storing the result in a host array
   */
  void compact_offsets_kernel_wrapper();

  /**
   * @brief This function is to check the correctness of the scan performed, called by gtests.
   * @param length
   * @param startoffset
   * @param stopoffset
   * @return A bool flag indicating whether the GPU and CPU results are consistent or not
   */
  bool check_correctness(int64_t length, int64_t startoffset, int64_t stopoffset);

};
}


