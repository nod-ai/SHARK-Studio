# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_VULKAN_SPIRV OR
   NOT IREE_HAL_DRIVER_VULKAN)
  message(STATUS "Missing Vulkan backend and/or driver, skipping vulkan_gui sample")
  return()
endif()

# This target statically links against Vulkan.
# One way to achieve this is by installing the Vulkan SDK from
# https://vulkan.lunarg.com/.
include(FindVulkan)
if(NOT Vulkan_FOUND)
  message(STATUS "Could not find Vulkan, skipping vulkan_gui sample")
  return()
endif()

# vcpkg install sdl2[vulkan]
#   tested with versions 2.0.14#4 - 2.0.22#1
find_package(SDL2)
if(NOT SDL2_FOUND)
  message(STATUS "Could not find SDL2, skipping vulkan_gui sample")
  return()
endif()

# Compile simple_mul.mlir to simple_mul.vmfb.
set(_COMPILE_TOOL_EXECUTABLE $<TARGET_FILE:iree-compile>)
set(_COMPILE_ARGS)
list(APPEND _COMPILE_ARGS "--iree-hal-target-backends=vulkan-spirv")
list(APPEND _COMPILE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/simple_mul.mlir")
list(APPEND _COMPILE_ARGS "-o")
list(APPEND _COMPILE_ARGS "simple_mul.vmfb")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.vmfb
  COMMAND ${_COMPILE_TOOL_EXECUTABLE} ${_COMPILE_ARGS}
  DEPENDS ${_COMPILE_TOOL_EXECUTABLE} "simple_mul.mlir"
)
# Embed simple_mul.vmfb into a C file as simple_mul_bytecode_module_c.[h/c]
set(_EMBED_DATA_EXECUTABLE $<TARGET_FILE:generate_embed_data>)
set(_EMBED_ARGS)
list(APPEND _EMBED_ARGS "--output_header=simple_mul_bytecode_module_c.h")
list(APPEND _EMBED_ARGS "--output_impl=simple_mul_bytecode_module_c.c")
list(APPEND _EMBED_ARGS "--identifier=iree_samples_vulkan_gui_simple_mul_bytecode_module")
list(APPEND _EMBED_ARGS "--flatten")
list(APPEND _EMBED_ARGS "${CMAKE_CURRENT_BINARY_DIR}/simple_mul.vmfb")
add_custom_command(
  OUTPUT "simple_mul_bytecode_module_c.h" "simple_mul_bytecode_module_c.c"
  COMMAND ${_EMBED_DATA_EXECUTABLE} ${_EMBED_ARGS}
  DEPENDS ${_EMBED_DATA_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.vmfb
)
# Define a library target for simple_mul_bytecode_module_c.
add_library(iree_samples_vulkan_gui_simple_mul_bytecode_module_c OBJECT)
target_sources(iree_samples_vulkan_gui_simple_mul_bytecode_module_c
  PRIVATE
    simple_mul_bytecode_module_c.h
    simple_mul_bytecode_module_c.c
)


FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui
  GIT_TAG        master
)

FetchContent_MakeAvailable(imgui)

# Dear ImGui
set(IMGUI_DIR ${CMAKE_BINARY_DIR}/_deps/imgui-src)
message("Looking for Imgui in ${IMGUI_DIR}")
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

# Define the sample executable.
set(_NAME "iree-samples-vulkan-gui")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    vulkan_inference_gui.cc
    "${IMGUI_DIR}/backends/imgui_impl_sdl.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
)
set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "iree-samples-vulkan-gui")
target_include_directories(${_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(${_NAME}
  SDL2::SDL2
  Vulkan::Vulkan
  #
  iree_runtime_runtime
  iree_base_internal_main
  iree_hal_drivers_vulkan_registration_registration
  iree_modules_hal_hal
  iree_samples_vulkan_gui_simple_mul_bytecode_module_c
  iree_vm_vm
  iree_vm_bytecode_module
  iree_vm_cc
)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(_GUI_LINKOPTS "-SUBSYSTEM:WINDOWS")
else()
  set(_GUI_LINKOPTS "")
endif()
target_link_options(${_NAME}
  PRIVATE
    ${_GUI_LINKOPTS}
)

message(STATUS "Configured vulkan_gui sample successfully")
