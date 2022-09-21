# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.23)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-samples C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# Core project dependency
#-------------------------------------------------------------------------------

message(STATUS "Fetching core IREE repo (this may take a few minutes)...")
# Note: for log output, set -DFETCHCONTENT_QUIET=OFF,
# see https://gitlab.kitware.com/cmake/cmake/-/issues/18238#note_440475

include(FetchContent)

FetchContent_Declare(
  iree
  GIT_REPOSITORY https://github.com/nod-ai/shark-runtime.git
  GIT_TAG a327e9e428293627afbd483dd62b54d49b342dc0 
  GIT_SUBMODULES_RECURSE OFF
  GIT_SHALLOW OFF
  GIT_PROGRESS ON
  USES_TERMINAL_DOWNLOAD ON
)

# Extend module path to find MLIR CMake modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")

# Disable core project features not needed for these out of tree samples.
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(iree)
FetchContent_GetProperties(iree SOURCE_DIR IREE_SOURCE_DIR)

#-------------------------------------------------------------------------------
# Individual samples
#-------------------------------------------------------------------------------

add_subdirectory(vulkan_gui)
