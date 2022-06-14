# RUN: %PYTHON %s
# XFAIL: *
from shark.shark_importer import SharkImporter

model_path = "https://tfhub.dev/google/lite-model/spice/1?lite-format=tflite"

# TODO: Currently unsupported:
# 1. Multiple unsupported dynamic operations (tfl.stride, range, gather).
# 2. Static version blocked by tfl.range not having a lowering for static fixed shapes.
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

# Traceback (most recent call last):
#   File "spice_test.py", line 20, in test_compile_tflite
#     self.compile_and_execute()
#   File "SHARK/tank/tflite/test_util.py", line 127, in compile_and_execute
#     iree_tflite_compile.compile_file(
#   File "SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/tflite.py", line 164, in compile_file
#     result = invoke_pipeline([import_cl, compile_cl])
#   File "SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/binaries.py", line 263, in invoke_pipeline
#     raise CompilerToolError(stage.completed)
# iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-import-tflite
# Diagnostics:
# SHARK/tank/tflite/tmp/spice_test.py/model.tflite:0:0: error: 'tfl.squeeze' op Squeeze: output type not ranked tensor
# SHARK/tank/tflite/tmp/spice_test.py/model.tflite:0:0: note: see current operation: %60 = "tfl.squeeze"(%59)

# Setting up for IREE compiler tflite
# Traceback (most recent call last):
#   File "spice.py", line 18, in <module>
#     my_shark_importer.compile()
#   File "SHARK/shark/shark_importer.py", line 96, in compile
#     self.shark_module.compile()
#   File "SHARK/shark/shark_inference.py", line 69, in compile
#     self.shark_runner = SharkRunner(self.model, self.input,
#   File "SHARK/shark/shark_runner.py", line 70, in __init__
#     self.model = ireec_tflite.compile_file(self.model,
#   File "SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/tflite.py", line 157, in compile_file
#     result = invoke_immediate(import_cl)
#   File "SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/binaries.py", line 200, in invoke_immediate
#     raise CompilerToolError(process)
# iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-import-tflite
# Diagnostics:
# SHARK/shark/tmp/spice.py/model.tflite:0:0: error: 'tfl.squeeze' op Squeeze: output type not ranked tensor
# SHARK/shark/tmp/spice.py/model.tflite:0:0: note: see current operation: %60 = "tfl.squeeze"(%59)