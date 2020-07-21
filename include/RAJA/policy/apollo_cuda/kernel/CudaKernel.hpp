/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with CUDA in an Apollo-managed context.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_apollo_cuda_kernel_CudaKernel_HPP
#define RAJA_policy_apollo_cuda_kernel_CudaKernel_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"


#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/kernel.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/policy/cuda/kernel/internal.hpp"

#include "apollo/Apollo.h"
#include "apollo/Region.h"

namespace RAJA {
namespace apollo_cuda {
namespace internal {

RAJA_INLINE
int getKernelNumRegs(const void *device_fptr)
{
    struct cudaFuncAttributes funcAttrib;
    cudaFuncGetAttributes(&funcAttrib, device_fptr);
    return funcAttrib.numRegs;
}

RAJA_INLINE
int getKernelMaxThreads(const void *device_fptr)
{
    struct cudaFuncAttributes funcAttrib;
    cudaFuncGetAttributes(&funcAttrib, device_fptr);
    return funcAttrib.maxThreadsPerBlock;
}


// NOTE[cdw]: Here we add getMaxThreads() and include a default
//            parameter for getMaxBlocks that allows the number of
//            threads to be overridden. This will facilitate
//            runtime explorations of num_threads with Apollo.

RAJA_INLINE
int getDeviceMaxThreads()
{
  static int max_threads = -1;

  if (max_threads <= 0) {
    int cur_device = -1;
    cudaGetDevice(&cur_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cur_device);
    int s_num_sm = prop.multiProcessorCount;
    int s_max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    max_threads = s_num_sm * s_max_threads_per_sm;
    // printf("MAX_THREADS=%d\n", max_threads);
  }

  return max_threads;
}

RAJA_INLINE
int getDeviceMaxBlocks(int threadsPerBlock=1024)
{
  static int max_blocks = -1;

  if (max_blocks <= 0) {
    int cur_device = -1;
    cudaGetDevice(&cur_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cur_device);
    int s_num_sm = prop.multiProcessorCount;
    int s_max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    max_blocks = s_num_sm * (s_max_threads_per_sm / threadsPerBlock);
    // printf("MAX_BLOCKS=%d\n", max_blocks);
  }

  return max_blocks;
}

/*! CUDA global function for launching Apollo kernels
 */


}  //end: namespace internal
}  //end: namespace apollo_cuda


/*!
 * CUDA kernel launch policy where Apollo is responsible for selecting
 * the number of physical blocks and threads.
 */
template <bool async0>
struct apollo_cuda_launch {};

// NOTE: We are still in the RAJA namespace at this point.
//
namespace statement
{
    /*!
     * RAJA::kernel statements that launch an apollo_cuda kernel.
     */
    template <typename... EnclosedStmts>
    using ApolloCudaKernel = CudaKernelExt<apollo_cuda_launch<false>, EnclosedStmts...>;

}  //end: namespace statement



namespace internal
{
/*!
 * Helper class specialization for Apollo to set the properties for
 * this CUDA kernel invocation.
 */
template<bool async0, typename StmtList, typename Data>
struct CudaLaunchHelper<apollo_cuda_launch<async0>,StmtList,Data>
{

    using executor_t = internal::cuda_statement_list_executor_t<StmtList, Data>;

    static constexpr bool async = async0;


    //NOTE: These functions are called by the CudaKernelExt class when this
    //      launch helper is passed to it.

    inline static void
    max_blocks_threads(
            int shmem_size,
            int &max_blocks,
            int &max_threads)
    {
        max_threads = 1024; // = RAJA::apollo_cuda::internal::getDeviceMaxThreads();
        max_blocks  = RAJA::apollo_cuda::internal::getDeviceMaxBlocks();

    }

    static void
    launch( Data const &data,
            internal::LaunchDims launch_dims,
            size_t shmem,
            cudaStream_t stream)
    {
        static cudaEvent_t     time_start;
        static cudaEvent_t     time_stop;
        static float           time_exec_ms = 0.0;
        static Apollo         *apollo             = Apollo::instance();
        static Apollo::Region *apolloRegion       = nullptr;

        // TODO[cdw]: (Per-kernel maxThreads and maxBlocks, not per-device)
        //            Extract the __global__ function ptr for the kernel and
        //            call the CUDA reflection API on it to calculate the actual
        //            compiled number of registers it is using, use this as a feature
        //            for learning.

        // TODO[cdw]: (Ensure distinct apolloRegions for different kernels)
        //            Verify that this code is being inlined for each unique kernel,
        //            and is not being re-used by multiple kernels.  If it is being re-used
        //            then we need to grab the apolloRegion* object from a map where the
        //            key is the __global__ fptr of the kernel being launched.

        static int max_threads = RAJA::apollo_cuda::internal::getDeviceMaxThreads();
        static int max_blocks  = RAJA::apollo_cuda::internal::getDeviceMaxBlocks();

        // TODO[cdw]: (Setup threads to be dynamically configurable?)
        static constexpr int num_threads  = 1024;
        int num_blocks   = launch_dims.num_blocks();

        // TODO[cdw]: (Extract the count of the number of elements from the data)
        //            Revisit this once things are working even without this,
        //            Apollo should find the "best on average" block size...
        int num_elements = 1;

        int policy_index = 0;
        int policy_count = 20;

        int block_size_opts[] = {   0,
                                   32,   64,  128,  192, 256,
                                  320,  384,  448,  512, 576,
                                  640,  704,  768,  832, 896,
                                  960, 1024, 2048, 4096};


        if (apolloRegion == nullptr) {
            // one-time initialization
            std::string code_location = apollo->getCallpathOffset();
            apolloRegion = new Apollo::Region(1,code_location.c_str(),20);
            cudaEventCreate(&time_start);
            cudaEventCreate(&time_stop);
        }
        auto func = internal::CudaKernelLauncherFixed<num_threads,Data,executor_t>;

        apolloRegion->begin();

        apolloRegion->setFeature((float)num_elements);
        policy_index = apolloRegion->getPolicyIndex();

        if (policy_index == 0) {
            num_blocks = RAJA::apollo_cuda::internal::getDeviceMaxBlocks();
        } else {
            num_blocks = block_size_opts[policy_index];
        }

        // TODO[cdw]: We're re-doing some work done by the CudaKernelExt statement,
        //            and for total efficiency should probably not.
        //            Revisit this for optimization once other things are working.
        launch_dims.blocks  = fitCudaDims(num_blocks,  launch_dims.blocks);


        cudaEventRecord(time_start, stream);
        /////
        //
        // Here we actually invoke the CUDA kernel:
        func<<<launch_dims.blocks, launch_dims.threads, shmem, stream>>>(data);
        //
        /////
        cudaEventRecord(time_stop, stream);


        cudaEventSynchronize(time_stop);
        cudaEventElapsedTime(&time_exec_ms, time_start, time_stop);
        apolloRegion->end(time_exec_ms);

        //printf("time_exec_ms=%0.12f\n", time_exec_ms);
    }
};



}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
