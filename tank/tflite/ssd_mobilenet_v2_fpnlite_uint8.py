# RUN: %PYTHON %s

from shark.shark_importer import SharkImporter
import os
import sys
import coco_data

model_path = "https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v2_fpnlite_dynamic_1.0_uint8.tflite"

# TODO: iree tflite dynamic input not supported.
def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)

    return [coco_data.generate_input(workdir, input_details)]


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
#   File "ssd_mobilenet_v2_fpnlite_uint8_test.py", line 37, in test_compile_tflite
#     self.compile_and_execute()
#   File " SHARK/tank/tflite/test_util.py", line 127, in compile_and_execute
#     iree_tflite_compile.compile_file(
#   File " SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/tflite.py", line 164, in compile_file
#     result = invoke_pipeline([import_cl, compile_cl])
#   File " SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/binaries.py", line 263, in invoke_pipeline
#     raise CompilerToolError(stage.completed)
# iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-import-tflite
# Diagnostics:
#  SHARK/tank/tflite/tmp/ssd_mobilenet_v2_fpnlite_uint8_test.py/model.tflite:0:0: error: 'tfl.resize_nearest_neighbor' op ConvertResizeOp: resize dynamic input not supported.
#  SHARK/tank/tflite/tmp/ssd_mobilenet_v2_fpnlite_uint8_test.py/model.tflite:0:0: note: see current operation: %411 = "tfl.resize_nearest_neighbor"

# Traceback (most recent call last):
#   File "ssd_mobilenet_v2_fpnlite_uint8.py", line 27, in <module>
#     my_shark_importer.compile()
#   File " SHARK/shark/shark_importer.py", line 96, in compile
#     self.shark_module.compile()
#   File " SHARK/shark/shark_inference.py", line 69, in compile
#     self.shark_runner = SharkRunner(self.model, self.input,
#   File " SHARK/shark/shark_runner.py", line 70, in __init__
#     self.model = ireec_tflite.compile_file(self.model,
#   File " SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/tflite.py", line 157, in compile_file
#     result = invoke_immediate(import_cl)
#   File " SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/binaries.py", line 200, in invoke_immediate
#     raise CompilerToolError(process)
# iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-import-tflite
# Diagnostics:
#  SHARK/shark/tmp/ssd_mobilenet_v2_fpnlite_uint8.py/model.tflite:0:0: error: 'tfl.resize_nearest_neighbor' op ConvertResizeOp: resize dynamic input not supported.
#  SHARK/shark/tmp/ssd_mobilenet_v2_fpnlite_uint8.py/model.tflite:0:0: note: see current operation: %411 = "tfl.resize_nearest_neighbor"(%350, %410) {align_corners = false, half_pixel_centers = false} :
