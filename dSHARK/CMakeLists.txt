project(lws-minimal-http-server-form-post-file C)
cmake_minimum_required(VERSION 2.8.12)
find_package(libwebsockets CONFIG REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${LWS_CMAKE_DIR})
message("${LWS_CMAKE_DIR}")
message("${LIBWEBSOCKETS_DEP_LIBS}")
include(CheckCSourceCompiles)
include(LwsCheckRequirements)

set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree_tools_iree-translate>)

# Define arguments passed to iree-translate
set(_ARGS)
list(APPEND _ARGS "-iree-input-type=mhlo")
list(APPEND _ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND _ARGS "-iree-hal-target-backends=cuda")
# Uncomment the line below to use vulkan-spirv backend
#list(APPEND _ARGS "-iree-hal-target-backends=vulkan-spirv")
list(APPEND _ARGS "${CMAKE_CURRENT_SOURCE_DIR}/simple_embedding_test.mlir")
list(APPEND _ARGS "-o")
list(APPEND _ARGS "simple_embedding_test_bytecode_module_cuda_c.vmfb")

# Translate MLIR file to VM bytecode module
add_custom_command(
  OUTPUT "simple_embedding_test_bytecode_module_cuda_c.vmfb"
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_ARGS}
  DEPENDS iree_tools_iree-translate
)


#-------------------------------------------------------------------------------
# Embedd the VM bytcode module into a c file via `generate_embed_data`.
#-------------------------------------------------------------------------------

# Define arguments passed to generate_embed_data
set(_ARGS)
list(APPEND _ARGS "--output_header=simple_embedding_test_bytecode_module_cuda_c.h")
list(APPEND _ARGS "--output_impl=simple_embedding_test_bytecode_module_cuda_c.c")
list(APPEND _ARGS "--identifier=simple_embedding_test_bytecode_module_cuda_c")
list(APPEND _ARGS "--flatten")
list(APPEND _ARGS "simple_embedding_test_bytecode_module_cuda_c.vmfb")

# Embed VM bytecode module into c source file
add_custom_command(
  OUTPUT
    "simple_embedding_test_bytecode_module_cuda_c.h"
    "simple_embedding_test_bytecode_module_cuda_c.c"
  COMMAND generate_embed_data ${_ARGS}
  DEPENDS generate_embed_data simple_embedding_test_bytecode_module_cuda_c.vmfb
)

add_library(simple_embedding_test_bytecode_module_cuda_c STATIC "")
target_sources(simple_embedding_test_bytecode_module_cuda_c
  PRIVATE
    simple_embedding_test_bytecode_module_cuda_c.c
    simple_embedding_test_bytecode_module_cuda_c.h
)

set(SAMP lws-minimal-http-server-form-post-file)
set(SRCS minimal-http-server-form-post-file.c)

set(requirements 1)
require_lws_config(LWS_ROLE_H1 1 requirements)
require_lws_config(LWS_WITH_SERVER 1 requirements)
require_lws_config(LWS_WITH_FILE_OPS 1 requirements)

if (requirements)
	add_executable(${SAMP} ${SRCS})

	if (websockets_shared)
		target_link_libraries(${SAMP} websockets_shared ${LIBWEBSOCKETS_DEP_LIBS} simple_embedding_test_bytecode_module_cuda_c iree_base_base iree_hal_hal iree_hal_cuda_registration_registration iree_modules_hal_hal iree_vm_vm iree_vm_bytecode_module)
		add_dependencies(${SAMP} websockets_shared)
	else()
		target_link_libraries(${SAMP} websockets ${LIBWEBSOCKETS_DEP_LIBS} simple_embedding_test_bytecode_module_cuda_c iree_base_base iree_hal_hal iree_hal_cuda_registration_registration iree_modules_hal_hal iree_vm_vm iree_vm_bytecode_module) 
	endif()
else()
	message("not requirements")
endif()
