#include <iostream>
#include <assert.h>
#include "include/awkward_offset_arrays.h"
int main() {
  int64_t length;
  for(int i = 0; i < 9; i++) {
    length =  pow(10, i);
    int64_t *tooffsets = new int64_t[length + 1];
    int32_t *fromstarts = new int32_t[length];
    int32_t *fromstops = new int32_t[length];

    int64_t startoffset = 0;
    int64_t stopoffset = 0;

    for (int64_t i = 0; i < length; i++) {
      fromstarts[i] = rand() % 300;
      fromstops[i] = fromstarts[i] + rand() % 300;
    }

    awkward::AwkwardOffsetArrayCuda<int64_t, int32_t> arr;
    arr.awkward_listarray_compact_offsets(&tooffsets, fromstarts, fromstops, startoffset, stopoffset, length);

    int *cpu_tooffsets = new int[length + 1];
    cpu_tooffsets[0] = 0;
    for (int i = 1; i <= length; i++) {
      cpu_tooffsets[i] = cpu_tooffsets[i - 1] + (fromstops[i - 1] - fromstarts[i - 1]);
    }
    bool flag = true;
    for (int i = 0; i <= length; i++) {
      if (cpu_tooffsets[i] != tooffsets[i]) {
        std::cout << "Element no matching " << i << " " << cpu_tooffsets[i] << " " << tooffsets[i] << std::endl;
        std::cout << "tooffsets status" << " " << fromstops[i - 1] - fromstarts[i - 1] << " " << tooffsets[i - 1]
                  << std::endl;
        flag = false;
      }
    }
    if (flag)
      std::cout << "Test Passed" << std::endl;
    else
      std::cout << "Test Failed" << std::endl;
  }


}


