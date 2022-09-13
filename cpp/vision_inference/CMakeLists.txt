# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  message(STATUS "Missing LLVM backend and/or embeddded elf loader, skipping vision_inference sample")
  return()
endif()

# vcpkg install stb
#   tested with version 2021-09-10
find_package(Stb)
if(NOT Stb_FOUND)
  message(STATUS "Could not find Stb, skipping vision inference sample")
  return()
endif()

# Compile mnist.mlir to mnist.vmfb.
set(_COMPILE_TOOL_EXECUTABLE $<TARGET_FILE:iree-compile>)
set(_COMPILE_ARGS)
list(APPEND _COMPILE_ARGS "--iree-input-type=mhlo")
list(APPEND _COMPILE_ARGS "--iree-hal-target-backends=llvm-cpu")
list(APPEND _COMPILE_ARGS "${IREE_SOURCE_DIR}/samples/models/mnist.mlir")
list(APPEND _COMPILE_ARGS "-o")
list(APPEND _COMPILE_ARGS "mnist.vmfb")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mnist.vmfb
  COMMAND ${_COMPILE_TOOL_EXECUTABLE} ${_COMPILE_ARGS}
  DEPENDS ${_COMPILE_TOOL_EXECUTABLE} "${IREE_SOURCE_DIR}/samples/models/mnist.mlir"
)
# Embed mnist.vmfb into a C file as mnist_bytecode_module_c.[h/c]
set(_EMBED_DATA_EXECUTABLE $<TARGET_FILE:generate_embed_data>)
set(_EMBED_ARGS)
list(APPEND _EMBED_ARGS "--output_header=mnist_bytecode_module_c.h")
list(APPEND _EMBED_ARGS "--output_impl=mnist_bytecode_module_c.c")
list(APPEND _EMBED_ARGS "--identifier=iree_samples_vision_inference_mnist_bytecode_module")
list(APPEND _EMBED_ARGS "--flatten")
list(APPEND _EMBED_ARGS "${CMAKE_CURRENT_BINARY_DIR}/mnist.vmfb")
add_custom_command(
  OUTPUT "mnist_bytecode_module_c.h" "mnist_bytecode_module_c.c"
  COMMAND ${_EMBED_DATA_EXECUTABLE} ${_EMBED_ARGS}
  DEPENDS ${_EMBED_DATA_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/mnist.vmfb
)
# Define a library target for mnist_bytecode_module_c.
add_library(iree_samples_vision_inference_mnist_bytecode_module_c OBJECT)
target_sources(iree_samples_vision_inference_mnist_bytecode_module_c
  PRIVATE
    mnist_bytecode_module_c.h
    mnist_bytecode_module_c.c
)

# Define the sample executable.
set(_NAME "iree-run-mnist-module")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    "image_util.h"
    "image_util.c"
    "iree-run-mnist-module.c"
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "iree-run-mnist-module")
target_include_directories(${_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_include_directories(${_NAME} PRIVATE
    ${Stb_INCLUDE_DIR}
)
target_link_libraries(${_NAME}
  iree_base_base
  iree_base_tracing
  iree_hal_hal
  iree_runtime_runtime
  iree_samples_vision_inference_mnist_bytecode_module_c
)

# Define a target that copies the test image into the build directory.
add_custom_target(iree_samples_vision_inference_test_image
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mnist_test.png" "${CMAKE_CURRENT_BINARY_DIR}/mnist_test.png")
add_dependencies(${_NAME} iree_samples_vision_inference_test_image)

message(STATUS "Configured vision_inference sample successfully")
