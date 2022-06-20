# RUN: %PYTHON %s

from shark.shark_importer import SharkImporter
import os
import sys
import coco_data

model_path = "https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v2_dynamic_1.0_int8.tflite"

# TODO: failed to legalize operation 'tosa.rescale'
def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)

    inputs = coco_data.generate_input(workdir, input_details)
    # Move input values from [0, 255] to [-128, 127].
    inputs = inputs - 128
    return [inputs]


if __name__ == '__main__':
    my_shark_importer = SharkImporter(model_path=model_path,
                                      model_type="tflite",
                                      model_source_hub="tfhub",
                                      device="cpu",
                                      dynamic=False,
                                      jit_trace=True)
    # Case1: Use default inputs
    my_shark_importer.compile()
    shark_results = my_shark_importer.forward()
    # Case2: Use manually set inputs
    input_details, output_details = my_shark_importer.get_model_details()
    inputs = generate_inputs(input_details)  # device_inputs
    my_shark_importer.compile(inputs)
    shark_results = my_shark_importer.forward(inputs)
    # print(shark_results)

# Traceback (most recent call last):
#   File "ssd_mobilenet_v2_int8_test.py", line 39, in test_compile_tflite
#     self.compile_and_execute()
#   File " SHARK/tank/tflite/test_util.py", line 127, in compile_and_execute
#     iree_tflite_compile.compile_file(
#   File "SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/tflite.py", line 164, in compile_file
#     result = invoke_pipeline([import_cl, compile_cl])
#   File "SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/binaries.py", line 263, in invoke_pipeline
#     raise CompilerToolError(stage.completed)
# iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-compile
# Diagnostics:
# SHARK/tank/tflite/tmp/ssd_mobilenet_v2_int8_test.py/model.tflite:0:0: error: failed to legalize operation 'tosa.rescale'
# compilation failed
#
#
# Invoked with:
#  iree-compile  SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/../_mlir_libs/iree-compile - --iree-input-type=tosa --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=dylib-llvm-aot -o= SHARK/tank/tflite/tmp/ssd_mobilenet_v2_int8_test.py/module.bytecode --iree-mlir-to-vm-bytecode-module --iree-llvm-embedded-linker-path= SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/../_mlir_libs/iree-lld --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false

# Traceback (most recent call last):
#   File "ssd_mobilenet_v2_int8.py", line 30, in <module>
#     my_shark_importer.compile()
#   File " SHARK/shark/shark_importer.py", line 96, in compile
#     self.shark_module.compile()
#   File " SHARK/shark/shark_inference.py", line 69, in compile
#     self.shark_runner = SharkRunner(self.model, self.input,
#   File " SHARK/shark/shark_runner.py", line 77, in __init__
#     ) = get_iree_compiled_module(self.model,
#   File " SHARK/shark/iree_utils.py", line 204, in get_iree_compiled_module
#     flatbuffer_blob = compile_module_to_flatbuffer(module, device, frontend,
#   File " SHARK/shark/iree_utils.py", line 175, in compile_module_to_flatbuffer
#     flatbuffer_blob = ireec.compile_str(
#   File " SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/core.py", line 293, in compile_str
#     result = invoke_immediate(cl, immediate_input=input_bytes)
#   File " SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/binaries.py", line 200, in invoke_immediate
#     raise CompilerToolError(process)
# iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-compile
# Diagnostics:
# SHARK/shark/tmp/ssd_mobilenet_v2_int8.py/model.tflite:0:0: error: failed to legalize operation 'tosa.rescale'
# compilation failed
# 
# 
# Invoked with:
#  iree-compile  SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/../_mlir_libs/iree-compile - --iree-input-type=tosa --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=dylib --iree-mlir-to-vm-bytecode-module --iree-llvm-embedded-linker-path= SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/../_mlir_libs/iree-lld --mlir-print-debuginfo --mlir-print-op-on-diagnostic=false -iree-llvm-target-triple=x86_64-linux-gnu
