# GSoC 2020 CERN Evaluation Task Anish Biswas(trickarcher)
This repository contains my solution to the Evaluation task given by [Jim Pivarski](https://github.com/jpivarski).

## The Task
The task is described as follows:

>For the evaluation task, the kernel I'd like you to translate is awkward_listarray_compact_offsets:
 ```c
 template <typename C, typename T>
ERROR awkward_listarray_compact_offsets(T* tooffsets, const C* fromstarts, const C* fromstops, int64_t startsoffset, int64_t stopsoffset, int64_t length) {
  tooffsets[0] = 0;
  for (int64_t i = 0;  i < length;  i++) {
    C start = fromstarts[startsoffset + i];
    C stop = fromstops[stopsoffset + i];
    if (stop < start) {
      return failure("stops[i] < starts[i]", i, kSliceNone);
    }
    tooffsets[i + 1] = tooffsets[i] + (stop - start);
  }
  return success();
}
```
>The key thing about this kernel is that it has a loop-carried dependency: the value of tooffsets[i + 1] depends on the value of tooffsets[i], and this implementation computes it sequentially. A naive translation of this to CUDA would require the thread with threadIdx.x == i + 1 to wait for threadIdx.x == i, which would defeat the purpose of GPU parallelism. 
>To solve these sorts of problems on a parallel device like a GPU, we resort to entirely different techniques, such as the Hillis-Steele scan; here is a tutorial for implementing it with CUDA.
>The evaluation task would be to apply the Hillis-Steele technique to the specific problem of replicating awkward_listarray_compact_offsets on the GPU, showing that it yields the same results in some test cases, and showing the speedup between CPU, a naive translation, and the Hillis-Steele-based implementation. The solution may involve multiple __global__ functions, one called after the other on the same data.
---
## Approach
* At first, I implemented the basic Hillis-Steele Scan which worked fine for upto 2048 elements. However, as the length of the array increased I could see the inherent flaws of using the basic approach.
One of the first things ofcourse is that the information held by `d_offsets` will be overriden if the threads are not synchronized properly. However, even if the threads are synchronized the problem of copying everything to a shared memory come into picture.
Since we are working with templates here, there's no guarantee on the size of the array elements which may lead us to copy elements whose size are much greater than the shared memory of a block. This is a very undesirable situation. However, even if I prevent this by keeping a check for sizes, the entire scan becomes very slow because
we would have to call `__syncthreads()` everytime we play with shared memories and wait for all the threads to finish. This becomes a problem when using multiple blocks since we can't parallelize more than the wrap size of the GPU.
* The Double-Buffered Version of the Sum Scan was what I coded up next. I loved the idea behind this approach. Basically, what happens here is that even if the threads attempt to re-write the `d_out` array, it's fine because now we have another array(`d_in`) which had the state of the `d_out` before it was operated on during the current step.
This allows us to skip the entire `__syncthreads()` stuff which leads to consistent code. Again, this approach comes with a obvious drawback. The drawback being the additional memory being consumed to keep track of two arrays. The problem worsens if you have array size in the order of `10^8`. However, I found this approach much better than using shared memory hence I went forward with this. 
* In my previous emails to Jim, we had discussed on a tool which would automatically convert the CPU Kernels to GPU Kernels. Keeping this in mind I created a seperate kernel called `compute_offsets` which was basically `(fromstops[i] - fromstarts[i])`. The idea behind this is that we could seperate whatever is required for a `CPU Kernel` to make it compatible for parallel prefix sum, reduce it to a `Parallel Sum Prefix Problem` and then conquer it using the various `scan` methods. If I was to put this in a fancy way, this would basically be `reduce and conquer` method of transforming `CPU Kernels` to `GPU Kernels`.
---
## Design

* The idea was simple i.e create a library which encompasses the compact_offsets functions. In a real project I would expect this function to be part of some bigger class something like `AwkwardOffsetListArray` which would contain the function.
---
## Table of Array Size vs GPU Scan Speed vs CPU Scan Speed
| Size of Array | Scan on GTX 1080 | Sequential Scan on i7 6900K 3.2 GHz |
| :---          | :---:            |    :---:                            |
| 1          | 0.00137 ms            |    0.00100 ms                            |
| 10          | 0.014272 ms            |    0.00200 ms                            |
| 1000          | 0.019552 ms           |    0.005000 ms                            |
| 10000          | 0.026176 ms           |    0.004000 ms                            |
| 100000          | 0.037088 ms           |    0.040000 ms                            |
| 1000000          | 0.058912 ms           |    0.339000 ms                            |
| 10000000          | 0.748448 ms           |    2 ms                           |
| 100000000          | 8.966336 ms           |    29 ms                        |
| 1000000000          | 104.456573 ms           |    289 ms                        |
---

(Note: The GPU Benchmarks exclude the time taken in transfer operations)

As evident, there's almost the GPU implementation is almost 3X faster than CPU at larger array sizes
---

## Build Instructions

### Linux
* Install CMake(version >= 3.8)
* `git submodule --init --recursive` to clone GoogleTests into ext/googletest
* `mkdir build && cd build`
* `cmake ..`
* `make`
* The example executable will be installed in the `build` folder 
* The tests executable will also be installed in the `build` folder

----
## Note
* The 1 test failing for offsets is intentional, I couldn't figure out a good `exit` strategy of what should be done if (stops < starts)
