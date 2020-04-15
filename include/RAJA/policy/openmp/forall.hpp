/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for OpenMP.
 *
 *          These methods should work on any platform that supports OpenMP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_openmp_HPP
#define RAJA_forall_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <iostream>
#include <type_traits>

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"


namespace RAJA
{

namespace policy
{
namespace omp
{
///
/// OpenMP parallel for policy implementation
///

template <typename Iterable, typename Func, typename InnerPolicy>
RAJA_INLINE void forall_impl(const omp_parallel_exec<InnerPolicy>& exec,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA::resources::Resource res{RAJA::resources::Host()};
  forall_impl(res, exec, iter, loop_body);
}
template <typename Iterable, typename Func, typename InnerPolicy>
RAJA_INLINE RAJA::resources::Event forall_impl(RAJA::resources::Resource &res,
                                               const omp_parallel_exec<InnerPolicy>&,
                                               Iterable&& iter,
                                               Func&& loop_body)
{
  RAJA::resources::Event event;

  RAJA::region<RAJA::omp_parallel_region>([&]() {
    using RAJA::internal::thread_privatize;
    auto body = thread_privatize(loop_body);
    event = forall_impl(res, InnerPolicy{}, iter, body.get_priv());
  });

  return event;
}

///
/// OpenMP for nowait policy implementation
///

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const omp_for_nowait_exec& exc,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA::resources::Resource res{RAJA::resources::Host()};
  forall_impl(res, exec, iter, loop_body);
}
template <typename Iterable, typename Func>
RAJA_INLINE RAJA::resources::Event forall_impl(RAJA::resources::Resource &res, 
                                               const omp_for_nowait_exec&,
                                               Iterable&& iter,
                                               Func&& loop_body)
{
  RAJA::resources::Host host_res = RAJA::resources::raja_get<RAJA::resources::Host>(res);

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for nowait
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    loop_body(begin_it[i]);
  }

  return host_res.get_event();
}

///
/// OpenMP parallel for policy implementation
///

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const omp_for_exec& exec,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA::resources::Resource res{RAJA::resources::Host()};
  forall_impl(res, exec, iter, loop_body);
}
template <typename Iterable, typename Func>
RAJA_INLINE RAJA::resources::Event forall_impl(RAJA::resources::Resource &res,
                                               const omp_for_exec&,
                                               Iterable&& iter,
                                               Func&& loop_body)
{
  RAJA::resources::Host host_res = RAJA::resources::raja_get<RAJA::resources::Host>(res);

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    loop_body(begin_it[i]);
  }

  return host_res.get_event();
}

///
/// OpenMP parallel for static policy implementation
///

template <typename Iterable, typename Func, size_t ChunkSize>
RAJA_INLINE void forall_impl(const omp_for_static<ChunkSize>& exec,
                             Iterable&& iter,
                             Func&& loop_body)
{
  RAJA::resources::Resource res{RAJA::resources::Host()};
  forall_impl(res, exec, iter, loop_body);
}
template <typename Iterable, typename Func, size_t ChunkSize>
RAJA_INLINE RAJA::resources::Event forall_impl(RAJA::resources::Resource &res,
                                               const omp_for_static<ChunkSize>&,
                                               Iterable&& iter,
                                               Func&& loop_body)
{
  RAJA::resources::Host host_res = RAJA::resources::raja_get<RAJA::resources::Host>(res);

  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for schedule(static, ChunkSize)
  for (decltype(distance_it) i = 0; i < distance_it; ++i) {
    loop_body(begin_it[i]);
  }

  return host_res.get_event();
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set
// segments using omp execution. Segment execution is defined by
// segment execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Iterate over index set segments using an omp parallel loop and
 *         segment dependency graph. Individual segment execution will use
 *         execution policy template parameter.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set.
 *
 ******************************************************************************
 */

/*
 * TODO: Fix this!!!
 */

/*
template <typename SEG_EXEC_POLICY_T, typename LOOP_BODY, typename ...
SEG_TYPES>
RAJA_INLINE void forall(
    ExecPolicy<omp_taskgraph_segit, SEG_EXEC_POLICY_T>,
    const IndexSet<SEG_TYPES ...>& iset,
    LOOP_BODY loop_body)
{
  if (!iset.dependencyGraphSet()) {
    std::cerr << "\n RAJA IndexSet dependency graph not set , "
              << "FILE: " << __FILE__ << " line: " << __LINE__ << std::endl;
    RAJA_ABORT_OR_THROW("IndexSet dependency graph");
  }

  IndexSet& ncis = (*const_cast<IndexSet*>(&iset));

  int num_seg = ncis.getNumSegments();

#pragma omp parallel for schedule(static, 1)
  for (int isi = 0; isi < num_seg; ++isi) {
    IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
    DepGraphNode* task = seg_info->getDepGraphNode();

    task->wait();

    executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

    task->reset();

    if (task->numDepTasks() != 0) {
      for (int ii = 0; ii < task->numDepTasks(); ++ii) {
        // Alternateively, we could get the return value of this call
        // and actively launch the task if we are the last depedent
        // task. In that case, we would not need the semaphore spin
        // loop above.
        int seg = task->depTaskNum(ii);
        DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
        dep->satisfyOne();
      }
    }

  }  // iterate over segments of index set
}
*/

}  // namespace omp

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
