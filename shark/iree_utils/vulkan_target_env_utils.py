# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict


def get_vulkan_target_env(vulkan_target_triple):
    arch, product, os = vulkan_target_triple.split("=")[1].split("-")
    triple = (arch, product, os)
    # get version
    version = get_version(triple=triple)
    # TODO get revision
    revision = 120

    # extensions
    extensions = get_extensions(triple)
    # get vendor
    vendor = get_vendor(triple)
    # get device type
    device_type = get_device_type(triple)
    # get capabilities
    capabilities = get_vulkan_target_capabilities(triple)
    target_env = f"#vk.target_env<{version}, r({revision}), {extensions}, {vendor}:{device_type}, #vk.caps< {capabilities} >>"
    return target_env


def get_vulkan_target_env_flag(vulkan_target_triple):
    target_env = get_vulkan_target_env(vulkan_target_triple)
    target_env_flag = f"--iree-vulkan-target-env={target_env}"
    return target_env_flag


def get_version(triple):
    arch, product, os = triple
    if os in ["android30", "android31"]:
        return "v1.1"
    if product in ["android30", "android31"]:
        return "v1.1"
    if arch in ["unknown"]:
        return "v1.1"
    return "v1.3"


def get_extensions(triple):
    def make_ext_list(ext_list):
        res = ""
        for e in ext_list:
            res += e + ", "
        res = f"[{res[:-2]}]"
        return res

    arch, product, os = triple
    if arch == "m1":
        ext = [
            "VK_KHR_16bit_storage",
            "VK_KHR_8bit_storage",
            "VK_KHR_shader_float16_int8",
            "VK_KHR_storage_buffer_storage_class",
            "VK_KHR_variable_pointers",
        ]
        return make_ext_list(ext_list=ext)

    if arch == "valhall":
        ext = [
            "VK_KHR_16bit_storage",
            "VK_KHR_8bit_storage",
            "VK_KHR_shader_float16_int8",
            "VK_KHR_spirv_1_4",
            "VK_KHR_storage_buffer_storage_class",
            "VK_KHR_variable_pointers",
        ]
        return make_ext_list(ext_list=ext)

    if arch == "adreno":
        ext = [
            "VK_KHR_16bit_storage",
            "VK_KHR_shader_float16_int8",
            "VK_KHR_spirv_1_4",
            "VK_KHR_storage_buffer_storage_class",
            "VK_KHR_variable_pointers",
        ]
        if os == "android31":
            ext.append("VK_KHR_8bit_storage")
        return make_ext_list(ext_list=ext)

    if get_vendor(triple) == "SwiftShader":
        ext = ["VK_KHR_storage_buffer_storage_class"]
        return make_ext_list(ext_list=ext)

    if arch == "unknown":
        ext = [
            "VK_KHR_storage_buffer_storage_class",
            "VK_KHR_variable_pointers",
        ]
        return make_ext_list(ext_list=ext)

    ext = [
        "VK_KHR_16bit_storage",
        "VK_KHR_8bit_storage",
        "VK_KHR_shader_float16_int8",
        "VK_KHR_spirv_1_4",
        "VK_KHR_storage_buffer_storage_class",
        "VK_KHR_variable_pointers",
        "VK_EXT_subgroup_size_control",
    ]

    if get_vendor(triple) == "NVIDIA" or arch == "rdna3":
        ext.append("VK_NV_cooperative_matrix")

    return make_ext_list(ext_list=ext)


def get_vendor(triple):
    arch, product, os = triple
    if arch == "unknown":
        return "Unknown"
    if arch in ["rdna1", "rdna2", "rdna3", "rgcn3", "rgcn4", "rgcn5"]:
        return "AMD"
    if arch == "valhall":
        return "ARM"
    if arch == "m1":
        return "Apple"
    if arch in ["turing", "ampere"]:
        return "NVIDIA"
    if arch == "ardeno":
        return "Qualcomm"
    if arch == "cpu":
        if product == "swiftshader":
            return "SwiftShader"
        return "Unknown"
    print(f"Vendor for target triple - {triple} not found. Using unknown")
    return "Unknown"


def get_device_type(triple):
    arch, product, _ = triple
    if arch == "unknown":
        return "Unknown"
    if arch == "cpu":
        return "CPU"
    if arch in ["turing", "ampere"]:
        return "DiscreteGPU"
    if arch in ["rdna1", "rdna2", "rdna3", "rgcn3", "rgcn5"]:
        if product == "ivega10":
            return "IntegratedGPU"
        return "DiscreteGPU"
    if arch in ["m1", "valhall", "adreno"]:
        return "IntegratedGPU"
    print(f"Device type for target triple - {triple} not found. Using unknown")
    return "Unknown"


# get all the capabilities for the device
# TODO: make a dataclass for capabilites and init using vulkaninfo
def get_vulkan_target_capabilities(triple):
    def get_subgroup_val(l):
        return int(sum([subgroup_feature[sgf] for sgf in l]))

    cap = OrderedDict()
    arch, product, os = triple
    subgroup_feature = {
        "Basic": 1,
        "Vote": 2,
        "Arithmetic": 4,
        "Ballot": 8,
        "Shuffle": 16,
        "ShuffleRelative": 32,
        "Clustered": 64,
        "Quad": 128,
        "PartitionedNV": 256,
    }
    cap["maxComputeSharedMemorySize"] = 16384
    cap["maxComputeWorkGroupInvocations"] = 128
    cap["maxComputeWorkGroupSize"] = [128, 128, 64]
    cap["subgroupSize"] = 32
    cap["subgroupFeatures"] = ["Basic"]
    cap["minSubgroupSize"] = None
    cap["maxSubgroupSize"] = None
    cap["shaderFloat16"] = False
    cap["shaderFloat64"] = False
    cap["shaderInt8"] = False
    cap["shaderInt16"] = False
    cap["shaderInt64"] = False
    cap["storageBuffer16BitAccess"] = False
    cap["storagePushConstant16"] = False
    cap["uniformAndStorageBuffer16BitAccess"] = False
    cap["storageBuffer8BitAccess"] = False
    cap["storagePushConstant8"] = False
    cap["uniformAndStorageBuffer8BitAccess"] = False
    cap["variablePointers"] = False
    cap["variablePointersStorageBuffer"] = False
    cap["coopmatCases"] = None

    if arch in ["rdna1", "rdna2", "rdna3"]:
        cap["maxComputeSharedMemorySize"] = 65536
        cap["maxComputeWorkGroupInvocations"] = 1024
        cap["maxComputeWorkGroupSize"] = [1024, 1024, 1024]

        cap["subgroupSize"] = 64
        cap["minSubgroupSize"] = 32
        cap["maxSubgroupSize"] = 64
        cap["subgroupFeatures"] = [
            "Basic",
            "Vote",
            "Arithmetic",
            "Ballot",
            "Shuffle",
            "ShuffleRelative",
            "Clustered",
            "Quad",
        ]

        cap["shaderFloat16"] = True
        cap["shaderFloat64"] = True
        cap["shaderInt8"] = True
        cap["shaderInt16"] = True
        cap["shaderInt64"] = True
        cap["storageBuffer16BitAccess"] = True
        cap["storagePushConstant16"] = True
        cap["uniformAndStorageBuffer16BitAccess"] = True
        cap["storageBuffer8BitAccess"] = True
        cap["storagePushConstant8"] = True
        cap["uniformAndStorageBuffer8BitAccess"] = True
        cap["variablePointers"] = True
        cap["variablePointersStorageBuffer"] = True

        if arch == "rdna3":
            # TODO: Get scope value
            cap["coopmatCases"] = [
                "mSize = 16, nSize = 16, kSize = 16, aType = f16, bType = f16, cType = f16, resultType = f16, scope = #vk.scope<Subgroup>"
            ]
        if product == "rx5700xt":
            cap["storagePushConstant16"] = False
            cap["storagePushConstant8"] = False

    elif arch in ["rgcn5", "rgcn4", "rgcn3"]:
        cap["maxComputeSharedMemorySize"] = 65536
        cap["maxComputeWorkGroupInvocations"] = 1024
        cap["maxComputeWorkGroupSize"] = [1024, 1024, 1024]

        cap["subgroupSize"] = 64
        cap["subgroupFeatures"] = [
            "Basic",
            "Vote",
            "Arithmetic",
            "Ballot",
            "Shuffle",
            "ShuffleRelative",
            "Clustered",
            "Quad",
        ]
        cap["minSubgroupSize"] = 64
        cap["maxSubgroupSize"] = 64

        if arch == "rgcn5":
            cap["shaderFloat16"] = True
            cap["shaderFloat64"] = True

            cap["storageBuffer16BitAccess"] = True

        cap["shaderInt8"] = True
        cap["shaderInt16"] = True
        cap["shaderInt64"] = True

        cap["storagePushConstant16"] = False
        cap["uniformAndStorageBuffer16BitAccess"] = True
        cap["storageBuffer8BitAccess"] = True
        cap["storagePushConstant8"] = False
        cap["uniformAndStorageBuffer8BitAccess"] = True

        cap["variablePointers"] = True
        cap["variablePointersStorageBuffer"] = True

    elif arch == "m1":
        cap["maxComputeSharedMemorySize"] = 32768
        cap["maxComputeWorkGroupInvocations"] = 1024
        cap["maxComputeWorkGroupSize"] = [1024, 1024, 1024]

        cap["subgroupSize"] = 32
        cap["subgroupFeatures"] = [
            "Basic",
            "Vote",
            "Arithmetic",
            "Ballot",
            "Shuffle",
            "ShuffleRelative",
            "Quad",
        ]

        cap["shaderFloat16"] = True
        cap["shaderFloat64"] = True
        cap["shaderInt8"] = True
        cap["shaderInt16"] = True
        cap["shaderInt64"] = True
        cap["storageBuffer16BitAccess"] = True
        cap["storagePushConstant16"] = True
        cap["uniformAndStorageBuffer16BitAccess"] = True
        cap["storageBuffer8BitAccess"] = True
        cap["storagePushConstant8"] = True
        cap["uniformAndStorageBuffer8BitAccess"] = True
        cap["variablePointers"] = True
        cap["variablePointersStorageBuffer"] = True

    elif arch == "valhall":
        cap["maxComputeSharedMemorySize"] = 32768
        cap["maxComputeWorkGroupInvocations"] = 512
        cap["maxComputeWorkGroupSize"] = [512, 512, 512]

        cap["subgroupSize"] = 16
        cap["subgroupFeatures"] = [
            "Basic",
            "Vote",
            "Arithmetic",
            "Ballot",
            "Clustered",
            "Quad",
        ]

        if os == "android31":
            cap["subgroupFeatures"].append("Shuffle")
            cap["subgroupFeatures"].append("ShuffleRelative")

        cap["shaderFloat16"] = True
        cap["shaderInt8"] = True
        cap["shaderInt16"] = True
        cap["storageBuffer16BitAccess"] = True
        cap["storagePushConstant16"] = True
        cap["uniformAndStorageBuffer16BitAccess"] = True
        cap["storageBuffer8BitAccess"] = True
        cap["storagePushConstant8"] = True
        cap["uniformAndStorageBuffer8BitAccess"] = True
        cap["variablePointers"] = True
        cap["variablePointersStorageBuffer"] = True

    elif arch == "cpu":
        if product == "swiftshader":
            cap["maxComputeSharedMemorySize"] = 16384
            cap["subgroupSize"] = 4
            cap["subgroupFeatures"] = [
                "Basic",
                "Vote",
                "Arithmetic",
                "Ballot",
                "Shuffle",
                "ShuffleRelative",
            ]

    elif arch in ["ampere", "turing"]:
        cap["maxComputeSharedMemorySize"] = 49152
        cap["maxComputeWorkGroupInvocations"] = 1024
        cap["maxComputeWorkGroupSize"] = [1024, 1024, 1024]

        cap["subgroupSize"] = 32
        cap["minSubgroupSize"] = 32
        cap["maxSubgroupSize"] = 32
        cap["subgroupFeatures"] = [
            "Basic",
            "Vote",
            "Arithmetic",
            "Ballot",
            "Shuffle",
            "ShuffleRelative",
            "Clustered",
            "Quad",
        ]

        cap["shaderFloat16"] = True
        cap["shaderFloat64"] = True
        cap["shaderInt8"] = True
        cap["shaderInt16"] = True
        cap["shaderInt64"] = True
        cap["storageBuffer16BitAccess"] = True
        cap["storagePushConstant16"] = True
        cap["uniformAndStorageBuffer16BitAccess"] = True
        cap["storageBuffer8BitAccess"] = True
        cap["storagePushConstant8"] = True
        cap["uniformAndStorageBuffer8BitAccess"] = True
        cap["variablePointers"] = True
        cap["variablePointersStorageBuffer"] = True

        cap["coopmatCases"] = [
            "mSize = 8, nSize = 8, kSize = 32, aType = i8, bType = i8, cType = i32, resultType = i32, scope = #vk.scope<Subgroup>",
            "mSize = 16, nSize = 16, kSize = 16, aType = f16, bType = f16, cType = f16, resultType = f16, scope = #vk.scope<Subgroup>",
            "mSize = 16, nSize = 16, kSize = 16, aType = f16, bType = f16, cType = f32, resultType = f32, scope = #vk.scope<Subgroup>",
        ]

    elif arch == "adreno":
        cap["maxComputeSharedMemorySize"] = 32768
        cap["maxComputeWorkGroupInvocations"] = 1024
        cap["maxComputeWorkGroupSize"] = [1024, 1024, 64]

        cap["subgroupSize"] = 64
        cap["subgroupFeatures"] = [
            "Basic",
            "Vote",
            "Arithmetic",
            "Ballot",
            "Shuffle",
            "ShuffleRelative",
            "Quad",
        ]

        cap["shaderFloat16"] = True
        cap["shaderInt8"] = True
        cap["shaderInt16"] = True

        cap["storageBuffer16BitAccess"] = True
        if os == "andorid31":
            cap["uniformAndStorageBuffer8BitAccess"] = True

        cap["variablePointers"] = True
        cap["variablePointersStorageBuffer"] = True

    elif arch == "unknown":
        cap["subgroupSize"] = 64
        cap["variablePointers"] = False
        cap["variablePointersStorageBuffer"] = False
    else:
        print(
            f"Architecture {arch} not matched. Using default vulkan target device capability"
        )

    def get_comma_sep_str(ele_list):
        l = ""
        for ele in ele_list:
            l += f"{ele}, "
        l = f"[{l[:-2]}]"
        return l

    res = ""
    for k, v in cap.items():
        if v is None or v == False:
            continue
        if isinstance(v, bool):
            res += f"{k} = {'unit' if v == True else None}, "
        elif isinstance(v, list):
            if k == "subgroupFeatures":
                res += f"subgroupFeatures = {get_subgroup_val(v)}: i32, "
            elif k == "maxComputeWorkGroupSize":
                res += f"maxComputeWorkGroupSize = dense<{get_comma_sep_str(v)}>: vector<{len(v)}xi32>, "
            elif k == "coopmatCases":
                cmc = ""
                for case in v:
                    cmc += f"#vk.coop_matrix_props<{case}>, "
                res += f"cooperativeMatrixPropertiesNV = [{cmc[:-2]}], "
            else:
                res += f"{k} = {get_comma_sep_str(v)}, "
        else:
            res += f"{k} = {v}, "
    res = res[:-2]
    return res
