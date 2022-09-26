// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan Graphics + IREE API Integration Sample.

#include <SDL.h>
#include <SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan.h>


#include <cstring>
#include <set>
#include <vector>
#include <fstream>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <utility>

#include "iree/hal/drivers/vulkan/api.h"

// IREE's C API:
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/registration/driver_module.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

// iree-run-module
#include "iree/base/internal/flags.h"
#include "iree/base/status_cc.h"
#include "iree/base/tracing.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/vm_util_cc.h"

// Other dependencies (helpers, etc.)
#include "iree/base/internal/main.h"

#define IMGUI_UNLIMITED_FRAME_RATE

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

IREE_FLAG(string, entry_function, "",
          "Name of a function contained in the module specified by module_file "
          "to run.");

// TODO(benvanik): move --function_input= flag into a util.
static iree_status_t parse_function_io(iree_string_view_t flag_name,
                                       void* storage,
                                       iree_string_view_t value) {
  auto* list = (std::vector<std::string>*)storage;
  list->push_back(std::string(value.data, value.size));
  return iree_ok_status();
}
static void print_function_io(iree_string_view_t flag_name, void* storage,
                              FILE* file) {
  auto* list = (std::vector<std::string>*)storage;
  if (list->empty()) {
    fprintf(file, "# --%.*s=\n", (int)flag_name.size, flag_name.data);
  } else {
    for (size_t i = 0; i < list->size(); ++i) {
      fprintf(file, "--%.*s=\"%s\"\n", (int)flag_name.size, flag_name.data,
              list->at(i).c_str());
    }
  }
}
static std::vector<std::string> FLAG_function_inputs;
IREE_FLAG_CALLBACK(
    parse_function_io, print_function_io, &FLAG_function_inputs, function_input,
    "An input (a) value or (b) buffer of the format:\n"
    "  (a) scalar value\n"
    "     value\n"
    "     e.g.: --function_input=\"3.14\"\n"
    "  (b) buffer:\n"
    "     [shape]xtype=[value]\n"
    "     e.g.: --function_input=\"2x2xi32=1 2 3 4\"\n"
    "Optionally, brackets may be used to separate the element values:\n"
    "  2x2xi32=[[1 2][3 4]]\n"
    "Raw binary files can be read to provide buffer contents:\n"
    "  2x2xi32=@some/file.bin\n"
    "numpy npy files (from numpy.save) can be read to provide 1+ values:\n"
    "  @some.npy\n"
    "Each occurrence of the flag indicates an input in the order they were\n"
    "specified on the command line.");

typedef struct iree_file_toc_t {
  const char* name;             // the file's original name
  char* data;             // beginning of the file
  size_t size;                  // length of the file
} iree_file_toc_t;

bool load_file(const char* filename, char** pOut, size_t* pSize)
{
    FILE* f = fopen(filename, "rb");
    if (f == NULL)
    {
        fprintf(stderr, "Can't open %s\n", filename);
        return false;
    }

    fseek(f, 0L, SEEK_END);
    *pSize = ftell(f);
    fseek(f, 0L, SEEK_SET);

    *pOut = (char*)malloc(*pSize);

    size_t size = fread(*pOut, *pSize, 1, f);

    fclose(f);

    return size != 0;
}

static VkAllocationCallbacks* g_Allocator = NULL;
static VkInstance g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_Device = VK_NULL_HANDLE;
static uint32_t g_QueueFamily = (uint32_t)-1;
static VkQueue g_Queue = VK_NULL_HANDLE;
static VkPipelineCache g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static uint32_t g_MinImageCount = 2;
static bool g_SwapChainRebuild = false;
static int g_SwapChainResizeWidth = 0;
static int g_SwapChainResizeHeight = 0;

static void check_vk_result(VkResult err) {
  if (err == 0) return;
  fprintf(stderr, "VkResult: %d\n", err);
  abort();
}

// Returns the names of the Vulkan layers used for the given IREE
// |extensibility_set| and |features|.
std::vector<const char*> GetIreeLayers(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features) {
  iree_host_size_t required_count;
  iree_hal_vulkan_query_extensibility_set(
      features, extensibility_set, /*string_capacity=*/0, &required_count,
      /*out_string_values=*/NULL);
  std::vector<const char*> layers(required_count);
  iree_hal_vulkan_query_extensibility_set(features, extensibility_set,
                                          layers.size(), &required_count,
                                          layers.data());
  return layers;
}

// Returns the names of the Vulkan extensions used for the given IREE
// |extensibility_set| and |features|.
std::vector<const char*> GetIreeExtensions(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features) {
  iree_host_size_t required_count;
  iree_hal_vulkan_query_extensibility_set(
      features, extensibility_set, /*string_capacity=*/0, &required_count,
      /*out_string_values=*/NULL);
  std::vector<const char*> extensions(required_count);
  iree_hal_vulkan_query_extensibility_set(features, extensibility_set,
                                          extensions.size(), &required_count,
                                          extensions.data());
  return extensions;
}

// Returns the names of the Vulkan extensions used for the given IREE
// |vulkan_features|.
std::vector<const char*> GetDeviceExtensions(
    VkPhysicalDevice physical_device,
    iree_hal_vulkan_features_t vulkan_features) {
  std::vector<const char*> iree_required_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED,
      vulkan_features);
  std::vector<const char*> iree_optional_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
      vulkan_features);

  uint32_t extension_count = 0;
  check_vk_result(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_count, nullptr));
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  check_vk_result(vkEnumerateDeviceExtensionProperties(
      physical_device, nullptr, &extension_count, extension_properties.data()));

  // Merge extensions lists, including optional and required for simplicity.
  std::set<const char*> ext_set;
  ext_set.insert("VK_KHR_swapchain");
  ext_set.insert(iree_required_extensions.begin(),
                 iree_required_extensions.end());
  for (int i = 0; i < iree_optional_extensions.size(); ++i) {
    const char* optional_extension = iree_optional_extensions[i];
    for (int j = 0; j < extension_count; ++j) {
      if (strcmp(optional_extension, extension_properties[j].extensionName) ==
          0) {
        ext_set.insert(optional_extension);
        break;
      }
    }
  }
  std::vector<const char*> extensions(ext_set.begin(), ext_set.end());
  return extensions;
}

std::vector<const char*> GetInstanceLayers(
    iree_hal_vulkan_features_t vulkan_features) {
  // Query the layers that IREE wants / needs.
  std::vector<const char*> required_layers = GetIreeLayers(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED, vulkan_features);
  std::vector<const char*> optional_layers = GetIreeLayers(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL, vulkan_features);

  // Query the layers that are available on the Vulkan ICD.
  uint32_t layer_property_count = 0;
  check_vk_result(
      vkEnumerateInstanceLayerProperties(&layer_property_count, NULL));
  std::vector<VkLayerProperties> layer_properties(layer_property_count);
  check_vk_result(vkEnumerateInstanceLayerProperties(&layer_property_count,
                                                     layer_properties.data()));

  // Match between optional/required and available layers.
  std::vector<const char*> layers;
  for (const char* layer_name : required_layers) {
    bool found = false;
    for (const auto& layer_property : layer_properties) {
      if (std::strcmp(layer_name, layer_property.layerName) == 0) {
        found = true;
        layers.push_back(layer_name);
        break;
      }
    }
    if (!found) {
      fprintf(stderr, "Required layer %s not available\n", layer_name);
      abort();
    }
  }
  for (const char* layer_name : optional_layers) {
    for (const auto& layer_property : layer_properties) {
      if (std::strcmp(layer_name, layer_property.layerName) == 0) {
        layers.push_back(layer_name);
        break;
      }
    }
  }

  return layers;
}

std::vector<const char*> GetInstanceExtensions(
    SDL_Window* window, iree_hal_vulkan_features_t vulkan_features) {
  // Ask SDL for its list of required instance extensions.
  uint32_t sdl_extensions_count = 0;
  SDL_Vulkan_GetInstanceExtensions(window, &sdl_extensions_count, NULL);
  std::vector<const char*> sdl_extensions(sdl_extensions_count);
  SDL_Vulkan_GetInstanceExtensions(window, &sdl_extensions_count,
                                   sdl_extensions.data());

  std::vector<const char*> iree_required_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED,
      vulkan_features);
  std::vector<const char*> iree_optional_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL,
      vulkan_features);

  // Merge extensions lists, including optional and required for simplicity.
  std::set<const char*> ext_set;
  ext_set.insert(sdl_extensions.begin(), sdl_extensions.end());
  ext_set.insert(iree_required_extensions.begin(),
                 iree_required_extensions.end());
  ext_set.insert(iree_optional_extensions.begin(),
                 iree_optional_extensions.end());
  std::vector<const char*> extensions(ext_set.begin(), ext_set.end());
  return extensions;
}

void SetupVulkan(iree_hal_vulkan_features_t vulkan_features,
                 const char** instance_layers, uint32_t instance_layers_count,
                 const char** instance_extensions,
                 uint32_t instance_extensions_count,
                 const VkAllocationCallbacks* allocator, VkInstance* instance,
                 uint32_t* queue_family_index,
                 VkPhysicalDevice* physical_device, VkQueue* queue,
                 VkDevice* device, VkDescriptorPool* descriptor_pool) {
  VkResult err;

  // Create Vulkan Instance
  {
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.enabledLayerCount = instance_layers_count;
    create_info.ppEnabledLayerNames = instance_layers;
    create_info.enabledExtensionCount = instance_extensions_count;
    create_info.ppEnabledExtensionNames = instance_extensions;
    err = vkCreateInstance(&create_info, allocator, instance);
    check_vk_result(err);
  }

  // Select GPU
  {
    uint32_t gpu_count;
    err = vkEnumeratePhysicalDevices(*instance, &gpu_count, NULL);
    check_vk_result(err);
    IM_ASSERT(gpu_count > 0);

    VkPhysicalDevice* gpus =
        (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * gpu_count);
    err = vkEnumeratePhysicalDevices(*instance, &gpu_count, gpus);
    check_vk_result(err);

    // Use the first reported GPU for simplicity.
    *physical_device = gpus[0];

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(*physical_device, &properties);
    fprintf(stdout, "Selected Vulkan device: '%s'\n", properties.deviceName);
    free(gpus);
  }

  // Select queue family. We want a single queue with graphics and compute for
  // simplicity, but we could also discover and use separate queues for each.
  {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(*physical_device, &count, NULL);
    VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(
        sizeof(VkQueueFamilyProperties) * count);
    vkGetPhysicalDeviceQueueFamilyProperties(*physical_device, &count, queues);
    for (uint32_t i = 0; i < count; i++) {
      if (queues[i].queueFlags &
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
        *queue_family_index = i;
        break;
      }
    }
    free(queues);
    IM_ASSERT(*queue_family_index != (uint32_t)-1);
  }

  // Create Logical Device (with 1 queue)
  {
    std::vector<const char*> device_extensions =
        GetDeviceExtensions(*physical_device, vulkan_features);
    const float queue_priority[] = {1.0f};
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = *queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = queue_priority;
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();

    // Enable timeline semaphores.
    VkPhysicalDeviceFeatures2 features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    create_info.pNext = &features2;
    VkPhysicalDeviceTimelineSemaphoreFeatures semaphore_features;
    memset(&semaphore_features, 0, sizeof(semaphore_features));
    semaphore_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    semaphore_features.pNext = features2.pNext;
    features2.pNext = &semaphore_features;
    semaphore_features.timelineSemaphore = VK_TRUE;

    err = vkCreateDevice(*physical_device, &create_info, allocator, device);
    check_vk_result(err);
    vkGetDeviceQueue(*device, *queue_family_index, 0, queue);
  }

  // Create Descriptor Pool
  {
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * IREE_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IREE_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    err =
        vkCreateDescriptorPool(*device, &pool_info, allocator, descriptor_pool);
    check_vk_result(err);
  }
}

void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd,
                       const VkAllocationCallbacks* allocator,
                       VkInstance instance, uint32_t queue_family_index,
                       VkPhysicalDevice physical_device, VkDevice device,
                       VkSurfaceKHR surface, int width, int height,
                       uint32_t min_image_count) {
  wd->Surface = surface;

  // Check for WSI support
  VkBool32 res;
  vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index,
                                       wd->Surface, &res);
  if (res != VK_TRUE) {
    fprintf(stderr, "Error no WSI support on physical device 0\n");
    exit(-1);
  }

  // Select Surface Format
  const VkFormat requestSurfaceImageFormat[] = {
      VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
  const VkColorSpaceKHR requestSurfaceColorSpace =
      VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
      physical_device, wd->Surface, requestSurfaceImageFormat,
      (size_t)IREE_ARRAYSIZE(requestSurfaceImageFormat),
      requestSurfaceColorSpace);

  // Select Present Mode
#ifdef IMGUI_UNLIMITED_FRAME_RATE
  VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR,
                                      VK_PRESENT_MODE_IMMEDIATE_KHR,
                                      VK_PRESENT_MODE_FIFO_KHR};
#else
  VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
  wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
      physical_device, wd->Surface, &present_modes[0],
      IREE_ARRAYSIZE(present_modes));

  // Create SwapChain, RenderPass, Framebuffer, etc.
  IM_ASSERT(min_image_count >= 2);
  ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical_device, device, wd,
                                         queue_family_index, allocator, width,
                                         height, min_image_count);

  // Set clear color.
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  memcpy(&wd->ClearValue.color.float32[0], &clear_color, 4 * sizeof(float));
}

void RenderFrame(ImGui_ImplVulkanH_Window* wd, VkDevice device, VkQueue queue) {
  VkResult err;

  VkSemaphore image_acquired_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  err = vkAcquireNextImageKHR(device, wd->Swapchain, UINT64_MAX,
                              image_acquired_semaphore, VK_NULL_HANDLE,
                              &wd->FrameIndex);
  check_vk_result(err);

  ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
  {
    err = vkWaitForFences(
        device, 1, &fd->Fence, VK_TRUE,
        UINT64_MAX);  // wait indefinitely instead of periodically checking
    check_vk_result(err);

    err = vkResetFences(device, 1, &fd->Fence);
    check_vk_result(err);
  }
  {
    err = vkResetCommandPool(device, fd->CommandPool, 0);
    check_vk_result(err);
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
    check_vk_result(err);
  }
  {
    VkRenderPassBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = wd->RenderPass;
    info.framebuffer = fd->Framebuffer;
    info.renderArea.extent.width = wd->Width;
    info.renderArea.extent.height = wd->Height;
    info.clearValueCount = 1;
    info.pClearValues = &wd->ClearValue;
    vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
  }

  // Record Imgui Draw Data and draw funcs into command buffer
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), fd->CommandBuffer);

  // Submit command buffer
  vkCmdEndRenderPass(fd->CommandBuffer);
  {
    VkPipelineStageFlags wait_stage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &image_acquired_semaphore;
    info.pWaitDstStageMask = &wait_stage;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &fd->CommandBuffer;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &render_complete_semaphore;

    err = vkEndCommandBuffer(fd->CommandBuffer);
    check_vk_result(err);
    err = vkQueueSubmit(queue, 1, &info, fd->Fence);
    check_vk_result(err);
  }
}

void PresentFrame(ImGui_ImplVulkanH_Window* wd, VkQueue queue) {
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  VkPresentInfoKHR info = {};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &render_complete_semaphore;
  info.swapchainCount = 1;
  info.pSwapchains = &wd->Swapchain;
  info.pImageIndices = &wd->FrameIndex;
  VkResult err = vkQueuePresentKHR(queue, &info);
  check_vk_result(err);
  wd->SemaphoreIndex =
      (wd->SemaphoreIndex + 1) %
      wd->ImageCount;  // Now we can use the next set of semaphores
}

static void CleanupVulkan() {
  vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

  vkDestroyDevice(g_Device, g_Allocator);
  vkDestroyInstance(g_Instance, g_Allocator);
}

static void CleanupVulkanWindow() {
  ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData,
                                  g_Allocator);
}

namespace iree {

extern "C" int iree_main(int argc, char** argv) {

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc > 1) {
    // Avoid iree-run-module spinning endlessly on stdin if the user uses single
    // dashes for flags.
    printf(
        "[ERROR] unexpected positional argument (expected none)."
        " Did you use pass a flag with a single dash ('-')?"
        " Use '--' instead.\n");
    return 1;
  }

  // --------------------------------------------------------------------------
  // Create a window.
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    fprintf(stderr, "Failed to initialize SDL\n");
    abort();
    return 1;
  }

  // Setup window
  // clang-format off
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(
      SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  // clang-format on
  SDL_Window* window = SDL_CreateWindow(
      "IREE Samples - Vulkan Inference GUI", SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
  if (window == nullptr)
  {
    const char* sdl_err = SDL_GetError();
    fprintf(stderr, "Error, SDL_CreateWindow returned: %s\n", sdl_err);
    abort();
    return 1;
  }

  // Setup Vulkan
  iree_hal_vulkan_features_t iree_vulkan_features =
      static_cast<iree_hal_vulkan_features_t>(
          IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS |
          IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS);
  std::vector<const char*> layers = GetInstanceLayers(iree_vulkan_features);
  std::vector<const char*> extensions =
      GetInstanceExtensions(window, iree_vulkan_features);
  SetupVulkan(iree_vulkan_features, layers.data(),
              static_cast<uint32_t>(layers.size()), extensions.data(),
              static_cast<uint32_t>(extensions.size()), g_Allocator,
              &g_Instance, &g_QueueFamily, &g_PhysicalDevice, &g_Queue,
              &g_Device, &g_DescriptorPool);

  // Create Window Surface
  VkSurfaceKHR surface;
  VkResult err;
  if (SDL_Vulkan_CreateSurface(window, g_Instance, &surface) == 0) {
    fprintf(stderr, "Failed to create Vulkan surface.\n");
    abort();
    return 1;
  }

  // Create Framebuffers
  int w, h;
  SDL_GetWindowSize(window, &w, &h);
  ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
  SetupVulkanWindow(wd, g_Allocator, g_Instance, g_QueueFamily,
                    g_PhysicalDevice, g_Device, surface, w, h, g_MinImageCount);

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;

  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplSDL2_InitForVulkan(window);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = g_Instance;
  init_info.PhysicalDevice = g_PhysicalDevice;
  init_info.Device = g_Device;
  init_info.QueueFamily = g_QueueFamily;
  init_info.Queue = g_Queue;
  init_info.PipelineCache = g_PipelineCache;
  init_info.DescriptorPool = g_DescriptorPool;
  init_info.Allocator = g_Allocator;
  init_info.MinImageCount = g_MinImageCount;
  init_info.ImageCount = wd->ImageCount;
  init_info.CheckVkResultFn = check_vk_result;
  ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);

  // Upload Fonts
  {
    // Use any command queue
    VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
    VkCommandBuffer command_buffer = wd->Frames[wd->FrameIndex].CommandBuffer;

    err = vkResetCommandPool(g_Device, command_pool, 0);
    check_vk_result(err);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(command_buffer, &begin_info);
    check_vk_result(err);

    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

    VkSubmitInfo end_info = {};
    end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    end_info.commandBufferCount = 1;
    end_info.pCommandBuffers = &command_buffer;
    err = vkEndCommandBuffer(command_buffer);
    check_vk_result(err);
    err = vkQueueSubmit(g_Queue, 1, &end_info, VK_NULL_HANDLE);
    check_vk_result(err);

    err = vkDeviceWaitIdle(g_Device);
    check_vk_result(err);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
  }

  // Demo state.
  bool show_iree_window = true;
  // --------------------------------------------------------------------------
  // Setup IREE.

  // Check API version.
  iree_api_version_t actual_version;
  iree_status_t status =
      iree_api_version_check(IREE_API_VERSION_LATEST, &actual_version);
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "IREE runtime API version: %d\n", actual_version);
  } else {
    fprintf(stderr, "Unsupported runtime API version: %d\n", actual_version);
    abort();
  }

  // Create a runtime Instance.
  iree_vm_instance_t* iree_instance = nullptr;
  IREE_CHECK_OK(
      iree_vm_instance_create(iree_allocator_system(), &iree_instance));

  // Register HAL drivers and VM module types.
  IREE_CHECK_OK(iree_hal_vulkan_driver_module_register(
      iree_hal_driver_registry_default()));
  IREE_CHECK_OK(iree_hal_module_register_all_types(iree_instance));

  // Create IREE Vulkan Driver and Device, sharing our VkInstance/VkDevice.
  fprintf(stdout, "Creating Vulkan driver/device\n");
  // Load symbols from our static `vkGetInstanceProcAddr` for IREE to use.
  iree_hal_vulkan_syms_t* iree_vk_syms = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_syms_create(
      reinterpret_cast<void*>(&vkGetInstanceProcAddr), iree_allocator_system(),
      &iree_vk_syms));
  // Create the driver sharing our VkInstance.
  iree_hal_driver_t* iree_vk_driver = nullptr;
  iree_string_view_t driver_identifier = iree_make_cstring_view("vulkan");
  iree_hal_vulkan_driver_options_t driver_options;
  driver_options.api_version = VK_API_VERSION_1_0;
  driver_options.requested_features = static_cast<iree_hal_vulkan_features_t>(
      IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS);
  IREE_CHECK_OK(iree_hal_vulkan_driver_create_using_instance(
      driver_identifier, &driver_options, iree_vk_syms, g_Instance,
      iree_allocator_system(), &iree_vk_driver));
  // Create a device sharing our VkDevice and queue.
  // We could also create a separate (possibly low priority) compute queue for
  // IREE, and/or provide a dedicated transfer queue.
  iree_string_view_t device_identifier = iree_make_cstring_view("vulkan");
  iree_hal_vulkan_queue_set_t compute_queue_set;
  compute_queue_set.queue_family_index = g_QueueFamily;
  compute_queue_set.queue_indices = 1 << 0;
  iree_hal_vulkan_queue_set_t transfer_queue_set;
  transfer_queue_set.queue_indices = 0;
  iree_hal_device_t* iree_vk_device = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_wrap_device(
      device_identifier, &driver_options.device_options, iree_vk_syms,
      g_Instance, g_PhysicalDevice, g_Device, &compute_queue_set,
      &transfer_queue_set, iree_allocator_system(), &iree_vk_device));
  // Create a HAL module using the HAL device.
  iree_vm_module_t* hal_module = nullptr;
  IREE_CHECK_OK(iree_hal_module_create(iree_instance, iree_vk_device,
                                       IREE_HAL_MODULE_FLAG_NONE,
                                       iree_allocator_system(), &hal_module));


  // Load bytecode module
  //iree_file_toc_t module_file_toc;
  //const char network_model[] = "resnet50_tf.vmfb";
  //fprintf(stdout, "Loading: %s\n", network_model);
  //if (load_file(network_model, &module_file_toc.data, &module_file_toc.size) == false)
  //{
  //    abort();
  //    return 1;
  //}
  //fprintf(stdout, "module size: %zu\n", module_file_toc.size);

  iree_vm_module_t* bytecode_module = nullptr;
  iree_status_t module_status = iree_tooling_load_module_from_flags(
      iree_instance, iree_allocator_system(), &bytecode_module);
  if (!iree_status_is_ok(module_status))
    return -1;
  //IREE_CHECK_OK(iree_vm_bytecode_module_create(
  //    iree_instance,
  //    iree_const_byte_span_t{
  //        reinterpret_cast<const uint8_t*>(module_file_toc.data),
  //        module_file_toc.size},
  //    iree_allocator_null(), iree_allocator_system(), &bytecode_module));
  //// Query for details about what is in the loaded module.
  //iree_vm_module_signature_t bytecode_module_signature =
  //    iree_vm_module_signature(bytecode_module);
  //fprintf(stdout, "Module loaded, have <%" PRIhsz "> exported functions:\n",
  //        bytecode_module_signature.export_function_count);
  //for (int i = 0; i < bytecode_module_signature.export_function_count; ++i) {
  //  iree_vm_function_t function;
  //  IREE_CHECK_OK(iree_vm_module_lookup_function_by_ordinal(
  //      bytecode_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function));
  //  auto function_name = iree_vm_function_name(&function);
  //  auto function_signature = iree_vm_function_signature(&function);

  //  fprintf(stdout, "  %d: '%.*s' with calling convention '%.*s'\n", i,
  //          (int)function_name.size, function_name.data,
  //          (int)function_signature.calling_convention.size,
  //          function_signature.calling_convention.data);
  //}

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* iree_context = nullptr;
  std::vector<iree_vm_module_t*> modules = {hal_module, bytecode_module};
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      iree_instance, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
      iree_allocator_system(), &iree_context));
  fprintf(stdout, "Context with modules is ready for use\n");

  // Lookup the entry point function.
  iree_vm_function_t main_function;
  const char kMainFunctionName[] = "module.forward";
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      iree_context,
      iree_string_view_t{kMainFunctionName, sizeof(kMainFunctionName) - 1},
      &main_function));
  iree_string_view_t main_function_name = iree_vm_function_name(&main_function);
  fprintf(stdout, "Resolved main function named '%.*s'\n",
          (int)main_function_name.size, main_function_name.data);

  // --------------------------------------------------------------------------

        // Write inputs into mappable buffers.
        iree_hal_allocator_t* allocator =
            iree_hal_device_allocator(iree_vk_device);
        //iree_hal_memory_type_t input_memory_type =
        //    static_cast<iree_hal_memory_type_t>(
        //        IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
        //        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
        //iree_hal_buffer_usage_t input_buffer_usage =
        //    static_cast<iree_hal_buffer_usage_t>(IREE_HAL_BUFFER_USAGE_DEFAULT);
        //iree_hal_buffer_params_t buffer_params;
        //buffer_params.type = input_memory_type;
        //buffer_params.usage = input_buffer_usage;
        //buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;

       // Wrap input buffers in buffer views.

        vm::ref<iree_vm_list_t> inputs;
        iree_status_t input_status = ParseToVariantList(
            allocator,
            iree::span<const std::string>{FLAG_function_inputs.data(),
                                          FLAG_function_inputs.size()},
            iree_allocator_system(), &inputs);
        if (!iree_status_is_ok(input_status))
            return -1;
        //vm::ref<iree_vm_list_t> inputs;
        //IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, 6, iree_allocator_system(), &inputs));

        //iree_hal_buffer_view_t* input0_buffer_view = nullptr;
        //constexpr iree_hal_dim_t input_buffer_shape[] = {1, 224, 224, 3};
        //IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer(
        //    allocator,
        //    /*shape_rank=*/4, /*shape=*/input_buffer_shape,
        //    IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        //    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
        //    iree_make_const_byte_span(&input_res50, sizeof(input_res50)),
        //    &input0_buffer_view));

        //auto input0_buffer_view_ref = iree_hal_buffer_view_move_ref(input0_buffer_view);
        //IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs.get(), &input0_buffer_view_ref));

        // Prepare outputs list to accept results from the invocation.

        vm::ref<iree_vm_list_t> outputs;
        constexpr iree_hal_dim_t kOutputCount = 1000;
        IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, kOutputCount * sizeof(float), iree_allocator_system(), &outputs));

  // --------------------------------------------------------------------------

  // Main loop.
  bool done = false;
  while (!done) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        done = true;
      }

      ImGui_ImplSDL2_ProcessEvent(&event);
      if (event.type == SDL_QUIT) done = true;
      if (event.type == SDL_WINDOWEVENT &&
          event.window.event == SDL_WINDOWEVENT_RESIZED &&
          event.window.windowID == SDL_GetWindowID(window)) {
        g_SwapChainResizeWidth = (int)event.window.data1;
        g_SwapChainResizeHeight = (int)event.window.data2;
        g_SwapChainRebuild = true;
      }
    }

    if (g_SwapChainRebuild) {
      g_SwapChainRebuild = false;
      ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
      ImGui_ImplVulkanH_CreateOrResizeWindow(
          g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData,
          g_QueueFamily, g_Allocator, g_SwapChainResizeWidth,
          g_SwapChainResizeHeight, g_MinImageCount);
      g_MainWindowData.FrameIndex = 0;
    }

    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame(window);
    ImGui::NewFrame();

    // Custom window.
    {
      ImGui::Begin("IREE Vulkan Integration Demo", &show_iree_window);

      ImGui::Separator();

      // ImGui Inputs for two input tensors.
      // Run computation whenever any of the values changes.
      static bool dirty = true;
      if (dirty) {

        // Synchronously invoke the function.
        IREE_CHECK_OK(iree_vm_invoke(iree_context, main_function,
                                     IREE_VM_INVOCATION_FLAG_NONE,
                                     /*policy=*/nullptr, inputs.get(),
                                     outputs.get(), iree_allocator_system()));


        // we want to run continuously so we can use tools like RenderDoc, RGP, etc...
        dirty = true;
      }

      // Framerate counter.
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

      ImGui::End();
    }

    // Rendering
    ImGui::Render();
    RenderFrame(wd, g_Device, g_Queue);

    PresentFrame(wd, g_Queue);
  }
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Cleanup
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);
  iree_vm_context_release(iree_context);
  iree_hal_device_release(iree_vk_device);
  iree_hal_allocator_release(allocator);
  iree_hal_driver_release(iree_vk_driver);
  iree_hal_vulkan_syms_release(iree_vk_syms);
  iree_vm_instance_release(iree_instance);

  err = vkDeviceWaitIdle(g_Device);
  check_vk_result(err);
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  CleanupVulkanWindow();
  CleanupVulkan();

  SDL_DestroyWindow(window);
  SDL_Quit();
  // --------------------------------------------------------------------------

  return 0;
}

}  // namespace iree
