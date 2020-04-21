//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing basic functional tests for atomic operations with forall and views.
///

#ifndef __TEST_FORALL_ATOMIC_VIEW_HPP__
#define __TEST_FORALL_ATOMIC_VIEW_HPP__

#include <RAJA/RAJA.hpp>
#include "RAJA_gtest.hpp"

TYPED_TEST_SUITE_P(SeqForallAtomicViewFunctionalTest);

template <typename T>
class SeqForallAtomicViewFunctionalTest : public ::testing::Test
{
};

#if defined(RAJA_ENABLE_CUDA)
TYPED_TEST_SUITE_P(CudaForallAtomicViewFunctionalTest);

template <typename T>
class CudaForallAtomicViewFunctionalTest : public ::testing::Test
{
};
#endif

#if defined(RAJA_ENABLE_HIP)
TYPED_TEST_SUITE_P(HipForallAtomicViewFunctionalTest);

template <typename T>
class HipForallAtomicViewFunctionalTest : public ::testing::Test
{
};
#endif

template <typename ExecPolicy,
          typename AtomicPolicy,
          typename WORKINGRES,
          typename T>
void testAtomicViewBasic( RAJA::Index_type N )
{
  RAJA::TypedRangeSegment<RAJA::Index_type> seg(0, N);
  RAJA::TypedRangeSegment<RAJA::Index_type> seg_half(0, N / 2);

  camp::resources::Resource src_res{WORKINGRES()};
  camp::resources::Resource dest_res{WORKINGRES()};

  T * source = src_res.allocate<T>(N);
  T * dest = dest_res.allocate<T>(N/2);

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif

  RAJA::forall<RAJA::seq_exec>(seg,
                               [=](RAJA::Index_type i) { source[i] = (T)1; });

  // use atomic add to reduce the array
  RAJA::View<T, RAJA::Layout<1>> vec_view(source, N);

  RAJA::View<T, RAJA::Layout<1>> sum_view(dest, N);
  auto sum_atomic_view = RAJA::make_atomic_view<AtomicPolicy>(sum_view);


  // Zero out dest using atomic view
  RAJA::forall<ExecPolicy>(seg_half, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i) = (T)0;
  });

  // Assign values to dest using atomic view
  RAJA::forall<ExecPolicy>(seg, [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {
    sum_atomic_view(i / 2) += vec_view(i);
  });

#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaDeviceSynchronize());
#endif

#if defined(RAJA_ENABLE_HIP)
  hipErrchk(hipDeviceSynchronize());
#endif

  for (RAJA::Index_type i = 0; i < N / 2; ++i) {
    EXPECT_EQ((T)2, dest[i]);
  }

  src_res.deallocate( source );
  dest_res.deallocate( dest );
}

TYPED_TEST_P(SeqForallAtomicViewFunctionalTest, seq_ForallAtomicViewFunctionalTest)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<3>>::type;
  testAtomicViewBasic<AExec, APol, ResType, DType>( 100000 );
}

REGISTER_TYPED_TEST_SUITE_P( SeqForallAtomicViewFunctionalTest,
                             seq_ForallAtomicViewFunctionalTest
                           );

#if defined(RAJA_ENABLE_CUDA)
GPU_TYPED_TEST_P(CudaForallAtomicViewFunctionalTest, cuda_ForallAtomicViewFunctionalTest)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<3>>::type;
  testAtomicViewBasic<AExec, APol, ResType, DType>( 100000 );
}

REGISTER_TYPED_TEST_SUITE_P( CudaForallAtomicViewFunctionalTest,
                             cuda_ForallAtomicViewFunctionalTest
                           );
#endif

#if defined(RAJA_ENABLE_HIP)
GPU_TYPED_TEST_P(HipForallAtomicViewFunctionalTest, hip_ForallAtomicViewFunctionalTest)
{
  using AExec   = typename camp::at<TypeParam, camp::num<0>>::type;
  using APol    = typename camp::at<TypeParam, camp::num<1>>::type;
  using ResType = typename camp::at<TypeParam, camp::num<2>>::type;
  using DType   = typename camp::at<TypeParam, camp::num<3>>::type;
  testAtomicViewBasic<AExec, APol, ResType, DType>( 100000 );
}

REGISTER_TYPED_TEST_SUITE_P( HipForallAtomicViewFunctionalTest,
                             hip_ForallAtomicViewFunctionalTest
                           );
#endif

#endif  //__TEST_FORALL_ATOMIC_VIEW_HPP__
