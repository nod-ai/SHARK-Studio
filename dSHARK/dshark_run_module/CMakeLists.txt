cmake_minimum_required(VERSION 3.13.4)

add_library(run_module run_module.h run_module.c dshark_driver_module.c)

target_link_libraries(run_module iree_base_base 
	iree_hal_hal 
	iree_hal_cuda_registration_registration 
	iree_modules_hal_hal 
	iree_vm_vm 
	iree_vm_bytecode_module 
	#iree_runtime_runtime 
	iree_hal_local_loaders_system_library_loader 
	iree_hal_local_sync_driver
	)

set_target_properties(run_module PROPERTIES LINKER_LANGUAGE C)
