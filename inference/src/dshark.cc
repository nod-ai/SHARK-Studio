// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/cuda/api.h"
#include "iree/hal/cuda/cuda_buffer.h"
#include "iree/hal/cuda/cuda_device.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/registration/driver_module.h"
#include "iree/hal/cuda/status_util.h"
#include "iree/hal/dylib/registration/driver_module.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_library_loader.h"
#include "iree/hal/local/loaders/vmvx_module_loader.h"
#include "iree/hal/local/sync_device.h"
#include "iree/hal/vmvx/registration/driver_module.h"
#include "iree/modules/hal/module.h"
#include "iree/tools/utils/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/context.h"
#include "iree/vm/ref_cc.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

// Several slightly modified iree tools

namespace iree {

Status ParseToVariantList(iree_hal_allocator_t *allocator,
                          iree::span<const std::string> input_strings,
                          iree_vm_list_t **out_list) {
  *out_list = NULL;
  vm::ref<iree_vm_list_t> variant_list;
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(/*element_type=*/nullptr, input_strings.size(),
                          iree_allocator_system(), &variant_list));
  for (size_t i = 0; i < input_strings.size(); ++i) {
    iree_string_view_t input_view = iree_string_view_trim(iree_make_string_view(
        input_strings[i].data(), input_strings[i].size()));
    bool has_equal =
        iree_string_view_find_char(input_view, '=', 0) != IREE_STRING_VIEW_NPOS;
    bool has_x =
        iree_string_view_find_char(input_view, 'x', 0) != IREE_STRING_VIEW_NPOS;
    if (has_equal || has_x) {
      // Buffer view (either just a shape or a shape=value) or buffer.
      bool is_storage_reference = iree_string_view_consume_prefix(
          &input_view, iree_make_cstring_view("&"));
      iree_hal_buffer_view_t *buffer_view = nullptr;
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_view_parse(input_view, allocator, &buffer_view),
          "parsing value '%.*s'", (int)input_view.size, input_view.data);
      if (is_storage_reference) {
        // Storage buffer reference; just take the storage for the buffer view -
        // it'll still have whatever contents were specified (or 0) but we'll
        // discard the metadata.
        auto buffer_ref = iree_hal_buffer_retain_ref(
            iree_hal_buffer_view_buffer(buffer_view));
        iree_hal_buffer_view_release(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_ref));
      } else {
        auto buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_view_ref));
      }
    } else {
      // Scalar.
      bool has_dot = iree_string_view_find_char(input_view, '.', 0) !=
                     IREE_STRING_VIEW_NPOS;
      iree_vm_value_t val;
      if (has_dot) {
        // Float.
        val = iree_vm_value_make_f32(0.0f);
        if (!iree_string_view_atof(input_view, &val.f32)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value '%.*s' as f32",
                                  (int)input_view.size, input_view.data);
        }
      } else {
        // Integer.
        val = iree_vm_value_make_i64(0);
        if (!iree_string_view_atoi_int64(input_view, &val.i64)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value '%.*s' as i64",
                                  (int)input_view.size, input_view.data);
        }
      }
      IREE_RETURN_IF_ERROR(iree_vm_list_push_value(variant_list.get(), &val));
    }
  }
  *out_list = variant_list.release();
  return OkStatus();
}

Status PrintVariantList(iree_vm_list_t *variant_list, size_t max_element_count,
                        std::ostream *os) {
  for (iree_host_size_t i = 0; i < iree_vm_list_size(variant_list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant(variant_list, i, &variant),
                         "variant %zu not present", i);

    *os << "result[" << i << "]: ";
    if (iree_vm_variant_is_value(variant)) {
      switch (variant.type.value_type) {
      case IREE_VM_VALUE_TYPE_I8:
        *os << "i8=" << variant.i8 << "\n";
        break;
      case IREE_VM_VALUE_TYPE_I16:
        *os << "i16=" << variant.i16 << "\n";
        break;
      case IREE_VM_VALUE_TYPE_I32:
        *os << "i32=" << variant.i32 << "\n";
        break;
      case IREE_VM_VALUE_TYPE_I64:
        *os << "i64=" << variant.i64 << "\n";
        break;
      case IREE_VM_VALUE_TYPE_F32:
        *os << "f32=" << variant.f32 << "\n";
        break;
      case IREE_VM_VALUE_TYPE_F64:
        *os << "f64=" << variant.f64 << "\n";
        break;
      default:
        *os << "?\n";
        break;
      }
    } else if (iree_vm_variant_is_ref(variant)) {
      iree_string_view_t type_name =
          iree_vm_ref_type_name(variant.type.ref_type);
      *os << std::string(type_name.data, type_name.size) << "\n";
      if (iree_hal_buffer_view_isa(variant.ref)) {
        auto *buffer_view = iree_hal_buffer_view_deref(variant.ref);
        std::string result_str(4096, '\0');
        iree_status_t status;
        do {
          iree_host_size_t actual_length = 0;
          status = iree_hal_buffer_view_format(buffer_view, max_element_count,
                                               result_str.size() + 1,
                                               &result_str[0], &actual_length);
          result_str.resize(actual_length);
        } while (iree_status_is_out_of_range(status));
        IREE_RETURN_IF_ERROR(status);
        *os << result_str << "\n";
      } else {
        // TODO(benvanik): a way for ref types to describe themselves.
        *os << "(no printer)\n";
      }
    } else {
      *os << "(null)\n";
    }
  }

  return OkStatus();
}

iree_status_t CreateDevice(iree_allocator_t host_allocator,
                           iree_hal_device_t **out_device,
                           const char *device_name) {

  iree_hal_driver_t *driver = nullptr;

  IREE_CHECK_OK(iree_hal_driver_registry_try_create_by_name(
      iree_hal_driver_registry_default(), iree_make_cstring_view(device_name),
      host_allocator, &driver));
  IREE_CHECK_OK(iree_hal_driver_create_default_device(driver, host_allocator,
                                                      out_device));
  iree_hal_driver_release(driver);

  return iree_ok_status();
}

iree_status_t
iree_hal_register_all_available_drivers(iree_hal_driver_registry_t *registry) {
  IREE_TRACE_ZONE_BEGIN(z0);

#if defined(IREE_HAL_HAVE_CUDA_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_driver_module_register(registry));
#endif // IREE_HAL_HAVE_CUDA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_dylib_driver_module_register(registry));
#endif // IREE_HAL_HAVE_DYLIB_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_dylib_sync_driver_module_register(registry));
#endif // IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VMVX_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vmvx_driver_module_register(registry));
#endif // IREE_HAL_HAVE_VMVX_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VMVX_SYNC_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vmvx_sync_driver_module_register(registry));
#endif // IREE_HAL_HAVE_VMVX_SYNC_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VULKAN_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_driver_module_register(registry));
#endif // IREE_HAL_HAVE_VULKAN_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_EXPERIMENTAL_ROCM_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_rocm_driver_module_register(registry));
#endif // IREE_HAL_HAVE_EXPERIMENTAL_ROCM_DRIVER_MODULE

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

} // namespace iree

namespace triton {
namespace backend {
namespace dshark {

//
// Backend that demonstrates the TRITONBACKEND API. This backend works
// for any model that has 1 input with any datatype and any shape and
// 1 output with the same shape and datatype as the input. The backend
// supports both batching and non-batching models.
//
// For each batch of requests, the backend returns the input tensor
// value in the output tensor.
//

/////////////

extern "C" {

// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error *TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend) {
  const char *cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Triton TRITONBACKEND API version: ") +
               std::to_string(api_version_major) + "." +
               std::to_string(api_version_minor))
                  .c_str());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("'") + name + "' TRITONBACKEND API version: " +
               std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
               std::to_string(TRITONBACKEND_API_VERSION_MINOR))
                  .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message *backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char *buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message,
                                                      &buffer, &byte_size));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("backend configuration:\n") + buffer).c_str());

  // This backend does not require any "global" state but as an
  // example create a string to demonstrate.
  std::string *state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void *>(state)));

  return nullptr; // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error *TRITONBACKEND_Finalize(TRITONBACKEND_Backend *backend) {
  // Delete the "global" state associated with the backend.
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string *state = reinterpret_cast<std::string *>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
                  .c_str());

  delete state;

  return nullptr; // success
}

} // extern "C"

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
public:
  static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model,
                                    ModelState **state);
  virtual ~ModelState() = default;

  TRITONSERVER_Error *
  LoadModel(const std::string &artifact_name, iree_hal_device_t **device,
            std::string *model_path, iree_vm_instance_t **instance,
            iree_vm_context_t **context, iree_vm_module_t **input_module,
            iree_vm_module_t **hal_module);

  // Name of the input and output tensor
  const std::string &InputTensorName() const { return input_name_; }
  const std::string &OutputTensorName() const { return output_name_; }
  const std::string &DeviceName() const { return device_name_; }

  // Datatype of the input and output tensor
  TRITONSERVER_DataType InputTensorDataType() const { return input_datatype_; }
  TRITONSERVER_DataType OutputTensorDataType() const {
    return output_datatype_;
  }

  // Shape of the input and output tensor as given in the model
  // configuration file. This shape will not include the batch
  // dimension (if the model has one).
  const std::vector<int64_t> &InputTensorNonBatchShape() const {
    return input_shape_;
  }
  const std::vector<int64_t> &OutputTensorNonBatchShape() const {
    return output_shape_;
  }

  // Shape of the input and output tensor, including the batch
  // dimension (if the model has one). This method cannot be called
  // until the model is completely loaded and initialized, including
  // all instances of the model. In practice, this means that backend
  // should only call it in TRITONBACKEND_ModelInstanceExecute.
  TRITONSERVER_Error *TensorShape(std::vector<int64_t> &shape);

  // Validate that this model is supported by this backend.
  TRITONSERVER_Error *ValidateModelConfig();

private:
  ModelState(TRITONBACKEND_Model *triton_model);

  std::string input_name_;
  std::string output_name_;
  std::string device_name_;

  TRITONSERVER_DataType input_datatype_;
  TRITONSERVER_DataType output_datatype_;

  bool shape_initialized_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> shape_;
};

ModelState::ModelState(TRITONBACKEND_Model *triton_model)
    : BackendModel(triton_model), shape_initialized_(false) {
  // Validate that the model's configuration matches what is supported
  // by this backend.

  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error *ModelState::Create(TRITONBACKEND_Model *triton_model,
                                       ModelState **state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException &ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr; // success
}

TRITONSERVER_Error *ModelState::LoadModel(const std::string &artifact_name,
                                          iree_hal_device_t **device,
                                          std::string *model_path,
                                          iree_vm_instance_t **instance,
                                          iree_vm_context_t **context,
                                          iree_vm_module_t **input_module,
                                          iree_vm_module_t **hal_module) {

  // register the correct driver.  cuda for gpu and dylib for cpu

  IREE_CHECK_OK(
      iree_hal_cuda_driver_module_register(iree_hal_driver_registry_default()));

  // Find the binary file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.vmfb").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.vmfb";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(exists, TRITONSERVER_ERROR_UNAVAILABLE,
                          std::string("unable to find '") + *model_path +
                              "' for model instance '" + Name() + "'");
  }

  // load in data from binary file

  iree_file_contents_t *flatbuffer_contents = NULL;

  IREE_LOG(INFO) << model_path->c_str();

  iree_file_read_contents(model_path->c_str(), iree_allocator_system(),
                          &flatbuffer_contents);

  IREE_CHECK_OK(iree_hal_module_register_types());

  // initialize instance

  IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), instance));

  // initialize and create bytecode_module

  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      flatbuffer_contents->const_buffer,
      iree_file_contents_deallocator(flatbuffer_contents),
      iree_allocator_system(), input_module));

  // create the device

  IREE_CHECK_OK(iree::CreateDevice(iree_allocator_system(), device, "dylib"));

  // declare and create the hal_module

  IREE_CHECK_OK(
      iree_hal_module_create(*device, iree_allocator_system(), hal_module));

  // create the context

  std::array<iree_vm_module_t *, 2> modules = {*hal_module, *input_module};
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      *instance, IREE_VM_CONTEXT_FLAG_NONE, modules.data(), modules.size(),
      iree_allocator_system(), context));

  return nullptr;
}

TRITONSERVER_Error *ModelState::ValidateModelConfig() {
  // If verbose logging is enabled, dump the model's configuration as
  // JSON into the console output.
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
  }

  // ModelConfig is the model configuration as a TritonJson
  // object. Use the TritonJson utilities to parse the JSON and
  // determine if the configuration is supported by this backend.
  common::TritonJson::Value inputs, outputs, device_types;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("instance_group", &device_types));

  common::TritonJson::Value input, output, device_type;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));
  RETURN_IF_ERROR(device_types.IndexAsObject(0, &device_type));

  // get the device kind

  const char *device_name;
  size_t device_name_len;
  RETURN_IF_ERROR(
      device_type.MemberAsString("kind", &device_name, &device_name_len));
  device_name_ = std::string(device_name);

  // Record the input and output name in the model state.
  const char *input_name;
  size_t input_name_len;
  RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
  input_name_ = std::string(input_name);

  const char *output_name;
  size_t output_name_len;
  RETURN_IF_ERROR(
      output.MemberAsString("name", &output_name, &output_name_len));
  output_name_ = std::string(output_name);

  // Input and output must have same datatype
  std::string input_dtype, output_dtype;
  RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));

  input_datatype_ = ModelConfigDataTypeToTritonServerDataType(input_dtype);
  output_datatype_ = ModelConfigDataTypeToTritonServerDataType(output_dtype);

  // Input and output must have same shape. Reshape is not supported
  // on either input or output so flag an error is the model
  // configuration uses it.
  triton::common::TritonJson::Value reshape;
  RETURN_ERROR_IF_TRUE(input.Find("reshape", &reshape),
                       TRITONSERVER_ERROR_UNSUPPORTED,
                       std::string("reshape not supported for input tensor"));
  RETURN_ERROR_IF_TRUE(output.Find("reshape", &reshape),
                       TRITONSERVER_ERROR_UNSUPPORTED,
                       std::string("reshape not supported for output tensor"));

  std::vector<int64_t> input_shape, output_shape;
  RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

  input_shape_ = input_shape;
  output_shape_ = output_shape;

  return nullptr; // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model) {

  const char *cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInitialize: ") + name +
               " (version " + std::to_string(version) + ")")
                  .c_str());
  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model. If anything goes wrong with initialization
  // of the model state then an error is returned and Triton will fail
  // to load the model.
  ModelState *model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
      model, reinterpret_cast<void *>(model_state)));

  return nullptr; // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState *model_state = reinterpret_cast<ModelState *>(vstate);
  delete model_state;

  return nullptr; // success
}

} // extern "C"

/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
public:
  static TRITONSERVER_Error *
  Create(ModelState *model_state,
         TRITONBACKEND_ModelInstance *triton_model_instance,
         ModelInstanceState **state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState *StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(TRITONBACKEND_Request **requests,
                       const uint32_t request_count);

private:
  ModelInstanceState(ModelState *model_state,
                     TRITONBACKEND_ModelInstance *triton_model_instance);

  void Execute(std::vector<TRITONBACKEND_Response *> *responses,
               TRITONBACKEND_Request **requests, const uint32_t response_count,
               const uint32_t request_count, iree_vm_list_t *input_tensors,
               iree_vm_list_t *output_tensors,
               std::vector<const char *> output_names,
               std::vector<std::string> output_dtypes,
               std::vector<std::vector<int64_t>> output_shapes);
  TRITONSERVER_Error *SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request **requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response *> *responses,
      BackendInputCollector *collector, std::vector<const char *> *input_names,
      iree_vm_list_t **input_tensors,
      std::vector<BackendMemory *> *input_memories, bool *cuda_copy);
  void InitializeRuntimeEnvironment(iree_vm_module_t **input_module,
                                    iree_vm_module_t **hal_module,
                                    iree_hal_device_t **device,
                                    iree_vm_instance_t **instance,
                                    iree_vm_context_t **context);

  enum device_name_code { GPU_KIND, CPU_KIND, UNKNOWN };

  device_name_code hashit(std::string const &inString);

  ModelState *model_state_;

  std::string model_path_;

  iree_hal_device_t *device_;
  iree_vm_module_t *input_module_;
  iree_vm_module_t *hal_module_;
  iree_vm_instance_t *instance_;
  iree_vm_context_t *context_;

  // Map from configuration name for an input to the index of
  // that input in the model.
  std::unordered_map<std::string, int> input_index_map_;

  // Map from configuration name for an output to the index of
  // that output in the model.
  std::unordered_map<std::string, int> output_index_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;

  // If the input to the tensor is a dictionary of tensors.
  bool is_dict_input_;
};

void ModelInstanceState::ProcessRequests(TRITONBACKEND_Request **requests,
                                         const uint32_t request_count) {
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() +
               " with " + std::to_string(request_count) + " requests")
                  .c_str());
  const int max_batch_size = model_state_->MaxBatchSize();

  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string("null request given to DShark backend for '" +
                          Name() + "'")
                  .c_str()));
      return;
    }
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response *> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response *response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  for (size_t i = 0; i < request_count; i++) {
    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input *input;
      TRITONSERVER_Error *err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t *shape;
        uint64_t test_size;
        err = TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape,
                                            nullptr, &test_size, nullptr);

        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count,
                                          all_response_failed, err);
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if (!all_response_failed) {
    if ((total_batch_size != 1) &&
        (total_batch_size > (size_t)max_batch_size)) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
          responses, request_count, all_response_failed,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string("batch size " + std::to_string(total_batch_size) +
                          " for '" + Name() + "', max allowed is " +
                          std::to_string(max_batch_size))
                  .c_str()));
    }
  }

  std::vector<const char *> input_names;
  iree_vm_list_t *input_tensors = nullptr;
  std::vector<BackendMemory *> input_memories;
  bool cuda_copy = false;
  std::unique_ptr<BackendInputCollector> collector;

  if (!all_response_failed) {
    collector.reset(new BackendInputCollector(
        requests, request_count, &responses,
        model_state_->TritonMemoryManager(), false, nullptr));

    ModelInstanceState::device_name_code code =
        hashit(model_state_->DeviceName());

    if (code == GPU_KIND) {
      iree_cuda_set_current_thread(device_);
    }

    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        SetInputTensors(total_batch_size, requests, request_count, &responses,
                        collector.get(), &input_names, &input_tensors,
                        &input_memories, &cuda_copy));
  }

  // Request to retrieve all model outputs. 'output_names' and
  // 'output_tensors' are parallel vectors and so must be kept in
  // sync.
  std::vector<const char *> output_names;
  std::vector<std::string> output_dtypes;
  std::vector<std::vector<int64_t>> output_dims;
  iree_vm_list_t *output_tensors;
  iree_vm_list_create(/*element_type=*/nullptr, 16, iree_allocator_system(),
                      &output_tensors);
  if (!all_response_failed) {
    triton::common::TritonJson::Value ios;
    TRITONSERVER_Error *err =
        model_state_->ModelConfig().MemberAsArray("output", &ios);
    if (err == nullptr) {
      for (size_t i = 0; i < ios.ArraySize(); i++) {
        triton::common::TritonJson::Value io;
        err = ios.IndexAsObject(i, &io);
        if (err != nullptr) {
          break;
        }

        // Use names from ModelConfig by reference since the model
        // config will persist longer than this inference execution.
        const char *io_name;
        size_t io_name_len;
        std::string io_dtype;
        std::vector<int64_t> io_shape;
        err = io.MemberAsString("name", &io_name, &io_name_len);
        if (err != nullptr) {
          break;
        }
        err = io.MemberAsString("data_type", &io_dtype);
        if (err != nullptr) {
          break;
        }
        backend::ParseShape(io, "dims", &io_shape);

        output_names.emplace_back(io_name);
        output_dtypes.emplace_back(io_dtype);
        output_dims.emplace_back(io_shape);
      }
    }

    if (err != nullptr) {
      RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count,
                                        all_response_failed, err);
      output_names.clear();
      output_dtypes.clear();
      output_dims.clear();
    }
  }

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  // Run...

  if (!all_response_failed) {
    Execute(&responses, requests, request_count, request_count, input_tensors,
            output_tensors, output_names, output_dtypes, output_dims);
  }

  // Free BackendMemory used for inputs
  for (BackendMemory *mem : input_memories) {
    if (mem != nullptr) {
      delete mem;
    }
  }
  input_memories.clear();

  [[maybe_unused]] bool invalid_index = false;
  int max_index = 3; // fix this

  if (!all_response_failed) {
    for (const auto &name : output_names) {
      int op_index = output_index_map_[name];
      if ((op_index < 0) || (op_index > max_index)) {
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
            responses, request_count, all_response_failed,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                std::string(
                    "The output " + std::string(name) +
                    " in the model configuration refers to an output index "
                    "which"
                    " doesn't exist. This model has " +
                    std::to_string(max_index + 1) + " outputs")
                    .c_str()));
        invalid_index = true;
        break;
      }
    }
  }

  for (auto &response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(TRITONBACKEND_ResponseSend(
                       response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                   "failed to send DShark backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto &request = requests[r];
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(
                     TritonModelInstance(), request,
                     (responses[r] != nullptr) /* success */, 0, 0, 0, 0),
                 "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                     TritonModelInstance(), total_batch_size, 0, 0, 0, 0),
                 "failed reporting batch request statistics");
  }

  // iree_hal_device_release(device_);
  // iree_vm_context_release(context_);
  // iree_vm_instance_release(instance_);
  // iree_vm_module_release(hal_module_);
  // iree_vm_module_release(input_module_);
}

ModelInstanceState::device_name_code
ModelInstanceState::hashit(std::string const &inString) {
  if (inString == "KIND_GPU")
    return GPU_KIND;
  if (inString == "KIND_CPU")
    return CPU_KIND;
  return UNKNOWN;
}

// format the input tensors into a buffer compatable with the triton collector
// and pass them to the collector

TRITONSERVER_Error *ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request **requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response *> *responses,
    BackendInputCollector *collector, std::vector<const char *> *input_names,
    iree_vm_list_t **input_tensors,
    std::vector<BackendMemory *> *input_memories, bool *cuda_copy) {
  const int max_batch_size = model_state_->MaxBatchSize();

  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  std::vector<std::string> input_strings;

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input *input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char *input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t *input_shape;
    uint32_t input_dims_count;
    uint64_t test_size_2;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        &test_size_2, nullptr));

    input_names->emplace_back(input_name);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(input_shape,
                                      input_shape + input_dims_count);
    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    // The input must be in contiguous CPU/GPU memory.
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> alloc_perference;

    bool using_cpu = true;
    if (using_cpu) {
      alloc_perference = {{TRITONSERVER_MEMORY_CPU_PINNED, 0},
                          {TRITONSERVER_MEMORY_CPU, 0}};
    } else {
      alloc_perference = {{TRITONSERVER_MEMORY_GPU, 0}};
    }

    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;

    const char *input_buffer;
    size_t batchn_byte_size;

    RETURN_IF_ERROR(collector->ProcessTensor(
        input_name, nullptr, 0, alloc_perference, &input_buffer,
        &batchn_byte_size, &memory_type, &memory_type_id));

    std::string tstr;

    RETURN_IF_ERROR(BufferAsTypedString(tstr, input_buffer, batchn_byte_size,
                                        input_datatype));

    std::string type_str;

    switch (input_datatype) {
    case TRITONSERVER_TYPE_FP16:
      type_str = "f16";
      break;
    case TRITONSERVER_TYPE_FP32:
      type_str = "f32";
      break;
    case TRITONSERVER_TYPE_FP64:
      type_str = "f64";
      break;
    case TRITONSERVER_TYPE_INT8:
      type_str = "i8";
      break;
    case TRITONSERVER_TYPE_INT16:
      type_str = "i16";
      break;
    case TRITONSERVER_TYPE_INT32:
      type_str = "i32";
      break;
    case TRITONSERVER_TYPE_INT64:
      type_str = "i64";
      break;
    default:
      type_str = "?";
      break;
    }

    std::string input_shape_str = "";

    for (uint32_t i = 0; i < input_dims_count; i++) {
      input_shape_str = input_shape_str + std::to_string(input_shape[i]) + "x";
    }

    tstr.erase(remove(tstr.begin(), tstr.end(), ','), tstr.end());

    tstr = input_shape_str + type_str + "=" + tstr;

    input_strings.push_back(tstr);
  }

  IREE_CHECK_OK(ParseToVariantList(
      iree_hal_device_allocator(device_),
      iree::span<const std::string>{input_strings.data(), input_strings.size()},
      input_tensors));

  // Finalize...
  *cuda_copy |= collector->Finalize();

  return nullptr;
}

// set up everyting iree needs to run

void ModelInstanceState::InitializeRuntimeEnvironment(
    iree_vm_module_t **input_module, iree_vm_module_t **hal_module,
    iree_hal_device_t **device, iree_vm_instance_t **instance,
    iree_vm_context_t **context) {

  // Find the binary file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.vmfb").
  std::string cc_model_filename = "model.vmfb";
  std::string model_path;

  model_path =
      JoinPath({model_state_->RepositoryPath(),
                std::to_string(model_state_->Version()), cc_model_filename});

  // load in data from binary file

  iree_file_contents_t *flatbuffer_contents = NULL;

  IREE_LOG(INFO) << model_path.c_str();

  iree_file_read_contents(model_path.c_str(), iree_allocator_system(),
                          &flatbuffer_contents);

  IREE_CHECK_OK(iree_hal_module_register_types());

  // initialize instance

  IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), instance));

  // initialize and create bytecode_module

  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      flatbuffer_contents->const_buffer,
      iree_file_contents_deallocator(flatbuffer_contents),
      iree_allocator_system(), input_module));

  // create the device

  const char *driver_identifier;

  ModelInstanceState::device_name_code code =
      hashit(model_state_->DeviceName());

  IREE_LOG(INFO) << model_state_->DeviceName();

  switch (code) {
  case GPU_KIND:
    driver_identifier = "cuda";
    break;
  case CPU_KIND:
    driver_identifier = "dylib";
    break;
  default:
    IREE_LOG(INFO) << "Unrecognized Driver Identifier: using cpu";
    driver_identifier = "dylib";
    break;
  }

  IREE_CHECK_OK(
      iree::CreateDevice(iree_allocator_system(), device, driver_identifier));

  // declare and create the hal_module

  IREE_CHECK_OK(
      iree_hal_module_create(*device, iree_allocator_system(), hal_module));

  // create the context

  std::array<iree_vm_module_t *, 2> modules = {*hal_module, *input_module};
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      *instance, IREE_VM_CONTEXT_FLAG_NONE, modules.data(), modules.size(),
      iree_allocator_system(), context));
}

// execute model using setup unputs and environment

void ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response *> *responses,
    TRITONBACKEND_Request **requests, const uint32_t response_count,
    const uint32_t request_count, iree_vm_list_t *input_tensors,
    iree_vm_list_t *output_tensors, std::vector<const char *> output_names,
    std::vector<std::string> output_dtypes,
    std::vector<std::vector<int64_t>> output_shapes) {

  iree_vm_function_t function;

  IREE_CHECK_OK(input_module_->get_function(input_module_->self,
                                            IREE_VM_FUNCTION_LINKAGE_EXPORT, 0,
                                            &function, NULL, NULL));

  IREE_CHECK_OK(iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                               /*policy=*/NULL, input_tensors, output_tensors,
                               iree_allocator_system()));

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t *ret_buffer_view =
      (iree_hal_buffer_view_t *)iree_vm_list_get_ref_deref(
          output_tensors, 0, iree_hal_buffer_view_get_descriptor());

  if (ret_buffer_view == NULL) {
    IREE_LOG(INFO) << "can't find return buffer view";
  }

  // I want to do this is a seperate function that calls after execute in
  // process tensors I'm doing it like this for now so I can get a demo running

  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      false, false /* pinned_enabled */, nullptr /* stream*/);

  uint32_t output_count;
  TRITONBACKEND_RequestOutputCount(requests[0], &output_count);

  for (iree_host_size_t i = 0; i < output_count; ++i) {

    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_CHECK_OK(iree_vm_list_get_variant(output_tensors, i, &variant));

    auto *buffer_view = iree_hal_buffer_view_deref(variant.ref);
    std::string result_str(4096, '\0');
    iree_status_t status;
    do {
      iree_host_size_t actual_length = 0;
      status = iree_hal_buffer_view_format(buffer_view, (size_t)1024,
                                           result_str.size() + 1,
                                           &result_str[0], &actual_length);
      result_str.resize(actual_length);
    } while (iree_status_is_out_of_range(status));
    IREE_CHECK_OK(status);

    iree_hal_buffer_mapping_t buffer_mapping;
    IREE_CHECK_OK(iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(buffer_view), IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_WHOLE_BUFFER, &buffer_mapping));

    char *result = reinterpret_cast<char *>(buffer_mapping.contents.data);

    TRITONSERVER_DataType output_datatype =
        ModelConfigDataTypeToTritonServerDataType(output_dtypes[i]);

    responder.ProcessTensor(output_names[i], output_datatype, output_shapes[i],
                            result, TRITONSERVER_MEMORY_CPU, 0);
  }

  iree_vm_list_release(input_tensors);
  iree_vm_list_release(output_tensors);
}

TRITONSERVER_Error *
ModelInstanceState::Create(ModelState *model_state,
                           TRITONBACKEND_ModelInstance *triton_model_instance,
                           ModelInstanceState **state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException &ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr; // success
}

ModelInstanceState::ModelInstanceState(
    ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state) {
  ModelInstanceState::device_name_code code =
      hashit(model_state_->DeviceName());

  // TODO: Make this actually work.  In the meantime just register gpu

  switch (code) {
  case GPU_KIND:
    iree_hal_cuda_driver_module_register(iree_hal_driver_registry_default());
    break;
  case CPU_KIND:
    iree_hal_dylib_driver_module_register(iree_hal_driver_registry_default());
    break;
  default:
    IREE_LOG(INFO) << "Unrecognized Driver Identifier: using cpu";
    iree_hal_dylib_driver_module_register(iree_hal_driver_registry_default());
    break;
  }

  InitializeRuntimeEnvironment(&input_module_, &hal_module_, &device_,
                               &instance_, &context_);

  // THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
  //  ArtifactFilename(), &device_, &model_path_, &instance_, &context_,
  //  &input_module_, &hal_module_));

  [[maybe_unused]] size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
    }
  }
}

ModelInstanceState::~ModelInstanceState() {
  iree_hal_device_release(device_);
  iree_vm_context_release(context_);
  iree_vm_instance_release(instance_);
  iree_vm_module_release(hal_module_);
  iree_vm_module_release(input_module_);
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance) {
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model *model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void *vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState *model_state = reinterpret_cast<ModelState *>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState *instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void *>(instance_state)));

  return nullptr; // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState *instance_state =
      reinterpret_cast<ModelInstanceState *>(vstate);
  delete instance_state;

  return nullptr; // success
}

} // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                   TRITONBACKEND_Request **requests,
                                   const uint32_t request_count) {
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState *instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void **>(&instance_state)));
  ModelState *model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("model ") + model_state->Name() + ", instance " +
               instance_state->Name() + ", executing " +
               std::to_string(request_count) + " requests")
                  .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.

  instance_state->ProcessRequests(requests, request_count);

  return nullptr; // success
}

} // extern "C"

} // namespace dshark
} // namespace backend
} // namespace triton
