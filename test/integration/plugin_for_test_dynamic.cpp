//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/util/PluginStrategy.hpp"

#include <exception>

class ExceptionPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
    throw std::runtime_error("preLaunch");
  }

  void postLaunch(RAJA::util::PluginContext RAJA_UNUSED_ARG(p)) {
  }
};

extern "C" RAJA::util::PluginStrategy *getPlugin()
{
  return new ExceptionPlugin;
}