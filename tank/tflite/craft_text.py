# RUN: %PYTHON %s
from shark.shark_importer import SharkImporter
model_path = "https://tfhub.dev/tulasiram58827/lite-model/craft-text-detector/dr/1?lite-format=tflite"

# Failure is due to dynamic shapes:
#
# Traceback (most recent call last):
#   File "craft_text.py", line 22, in test_compile_tflite
#     self.compile_and_execute()
#   File "SHARK/tank/tflite/test_util.py", line 127, in compile_and_execute
#     iree_tflite_compile.compile_file(
#   File " SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/tflite.py", line 164, in compile_file
#     result = invoke_pipeline([import_cl, compile_cl])
#   File " SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/binaries.py", line 263, in invoke_pipeline
#     raise CompilerToolError(stage.completed)
# iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-import-tflite
# Diagnostics:
#  SHARK/tank/tflite/tmp/craft_text.py/model.tflite:0:0: error: 'tfl.resize_bilinear' op ConvertResizeOp: resize dynamic output not supported.
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
    # print(shark_results)