# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import coco_test_data
import numpy
import test_util

model_path = "https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v2_fpnlite_dynamic_1.0_float.tflite"

class SsdMobilenetV2FpnliteTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(SsdMobilenetV2FpnliteTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(SsdMobilenetV2FpnliteTest, self).compare_results(iree_results, tflite_results, details)
    for i in range(len(iree_results)):
      self.assertTrue(numpy.isclose(iree_results[i], tflite_results[i], atol=1e-4).all())

  def generate_inputs(self, input_details):
    inputs = coco_test_data.generate_input(self.workdir, input_details)
    # Normalize inputs to [-1, 1].
    inputs = (inputs.astype('float32') / 127.5) - 1
    return [inputs]

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

