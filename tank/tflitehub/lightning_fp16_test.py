# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"

# Currently failing further in the linalg stack:
#   Bug related to linalg fusion. Collapsing dimension despite linalg index.
class LightningFp16Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(LightningFp16Test, self).__init__(model_path, *args, **kwargs)

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()
