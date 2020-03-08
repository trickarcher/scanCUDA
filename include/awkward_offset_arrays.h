#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#pragma once

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

  T *d_in;
  T *d_out;
  C *fromstarts;
  C *fromstops;
  T *tooffsets;

  C *d_fromstarts;
  C *d_fromstops;
  T *d_tooffsets;

  int64_t startoffset;
  int64_t stopoffset;

  int64_t SIZE_C;
  int64_t SIZE_T;

  dim3 threads_per_block;
  dim3 blocks_per_grid;

 public:
  AwkwardOffsetArrayCuda<T, C>();

  void awkward_listarray_compact_offsets(T **tooffsets,
                                         C *fromstarts,
                                         C *fromstops,
                                         int64_t startsoffset,
                                         int64_t stopsoffset,
                                         int64_t length);

  void compute_offsets_kernel_wrapper();

  void compact_offsets_kernel_wrapper();

};
}


