#include <iostream>
#include "include/awkward_offset_arrays.h"

int main() {
  int64_t length;
  length = 1000000;
  int64_t *tooffsets = new int64_t[length + 1];
  int32_t *fromstarts = new int32_t[length];
  int32_t *fromstops = new int32_t[length];

  int64_t startoffset = 0;
  int64_t stopoffset = 0;

  auto max_limit = std::numeric_limits<int32_t>::max();
  srand(time(0));

  for (int64_t i = 0; i < length; i++) {
    int32_t val_1 = rand() % max_limit;
    int32_t val_2 = rand() % max_limit;
    fromstarts[i] = (val_1 <= val_2) ? val_1 : val_2;
    fromstops[i] = (val_1 >= val_2) ? val_1 : val_2;
  }

  awkward::AwkwardOffsetArrayCuda<int64_t, int32_t> arr;
  arr.awkward_listarray_compact_offsets(&tooffsets, fromstarts, fromstops, startoffset, stopoffset, length);

//  for(auto i = 0; i <=length; i++)
//  {
//    std::cout << tooffsets[i] << " ";
//  }
//  std::cout << "\n";
}


