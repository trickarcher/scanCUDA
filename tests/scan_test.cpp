#include "gtest/gtest.h"
#include "awkward_offset_arrays.h"

TEST(ScanTest, BasicSizeCheck) {
  awkward::AwkwardOffsetArrayCuda<int64_t, int8_t> arr;
  bool test_1 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(10, 0, 0);
  bool test_2 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(100, 0, 0);
  bool test_3 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(1000, 0, 0);
  bool test_4 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(100000, 0, 0);
  bool test_5 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(10000000, 0, 0);

  EXPECT_EQ (test_1, true);
  EXPECT_EQ (test_2, true);
  EXPECT_EQ (test_3, true);
  EXPECT_EQ (test_4, true);
  EXPECT_EQ (test_5, true);
}

TEST(ScanTest, OffsetCheck){
  awkward::AwkwardOffsetArrayCuda<int64_t, int8_t> arr;
  bool test_1 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(10, 1, 1);
  bool test_2 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(100, 10, 10);
  bool test_3 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(1000, 34, 34);
  bool test_4 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(100000, 123, 123);
  bool test_5 = arr.awkward::AwkwardOffsetArrayCuda<int64_t, int8_t>::check_correctness(10000000, 100, 200);

  EXPECT_EQ (test_1, true);
  EXPECT_EQ (test_2, true);
  EXPECT_EQ (test_3, true);
  EXPECT_EQ (test_4, true);
  EXPECT_EQ (test_5, true);
}

