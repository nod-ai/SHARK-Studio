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


function(iree_vulkan_sample)

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS"
    ${ARGN}
  )


  # Define the sample executable.
  set(_NAME "${_RULE_NAME}")
  set(SRCS "${_RULE_SRCS}")
  add_executable(${_NAME} "")
  target_sources(${_NAME}
    PRIVATE
      ${SRCS}
      "${IMGUI_DIR}/backends/imgui_impl_sdl.cpp"
      "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
      "${IMGUI_DIR}/imgui.cpp"
      "${IMGUI_DIR}/imgui_draw.cpp"
      "${IMGUI_DIR}/imgui_demo.cpp"
      "${IMGUI_DIR}/imgui_tables.cpp"
      "${IMGUI_DIR}/imgui_widgets.cpp"
  )
  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_NAME}")
  target_include_directories(${_NAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
  target_link_libraries(${_NAME}
    SDL2::SDL2
    Vulkan::Vulkan
    iree_runtime_runtime
    iree_base_internal_main
    iree_hal_drivers_vulkan_registration_registration
    iree_modules_hal_hal
    iree_vm_vm
    iree_vm_bytecode_module
    iree_vm_cc
    iree_tooling_vm_util_cc
    iree_tooling_context_util
  )

  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(_GUI_LINKOPTS "-SUBSYSTEM:CONSOLE")
  else()
    set(_GUI_LINKOPTS "")
  endif()

  target_link_options(${_NAME}
    PRIVATE
      ${_GUI_LINKOPTS}
  )
endfunction()

iree_vulkan_sample(
    NAME
      iree-samples-resnet-vulkan-gui

    SRCS
      vulkan_resnet_inference_gui.cc
)

iree_vulkan_sample(
    NAME
      iree-vulkan-gui

    SRCS
      vulkan_inference_gui.cc
)

message(STATUS "Configured vulkan_gui sample successfully")
