# RUN: %PYTHON %s
from shark.shark_importer import SharkImporter

model_path = "https://tfhub.dev/neso613/lite-model/ASR_TFLite/pre_trained_models/English/1?lite-format=tflite"


# Failure is due to dynamic shapes:
# - Some improvements to tfl.strided_slice lowering are next steps
#
#   File "SHARK/shark/shark_runner.py", line 70, in __init__
#     self.model = ireec_tflite.compile_file(self.model,
#   File "SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/tflite.py", line 157, in compile_file
#     result = invoke_immediate(import_cl)
#   File "SHARK/shark.venv/lib/python3.8/site-packages/iree/compiler/tools/binaries.py", line 200, in invoke_immediate
#     raise CompilerToolError(process)
# iree.compiler.tools.binaries.CompilerToolError: Error invoking IREE compiler tool iree-import-tflite
# Diagnostics:
# SHARK/shark/tmp/asr_conformer.py/model.tflite:0:0:
#   error: 'tfl.gather' op N, K, W, or C was calculated as <= zero.  Invalid dimensions for Gather
# SHARK/shark/tmp/asr_conformer.py/model.tflite:0:0:
#   note: see current operation: %480 = "tfl.gather"(%479, %477) {axis = 0 : i32, batch_dims = 0 : i32} :
#   (tensor<?x?xf32>, tensor<?x5xi32>) -> tensor<?x5x?xf32> loc("stft/frame/GatherV2;conformer_greedy/TensorArrayV2Write/TensorListSetItem"(
#   "SHARK/shark/tmp/asr_conformer.py/model.tflite":0:0))
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
