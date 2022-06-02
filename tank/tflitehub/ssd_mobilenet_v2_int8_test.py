# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import coco_test_data
import numpy
import test_util

model_path = "https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v2_dynamic_1.0_int8.tflite"

class SsdMobilenetV2Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(SsdMobilenetV2Test, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(SsdMobilenetV2Test, self).compare_results(iree_results, tflite_results, details)
    for i in range(len(iree_results)):
      # Dequantize outputs.
      zero_point = details[i]['quantization_parameters']['zero_points'][0]
      scale = details[i]['quantization_parameters']['scales'][0]
      dequantized_iree_results = (iree_results[i] - zero_point) * scale
      dequantized_tflite_results = (tflite_results[i] - zero_point) * scale
      self.assertTrue(numpy.isclose(dequantized_iree_results, dequantized_tflite_results, atol=0.1).all())

  def generate_inputs(self, input_details):
    inputs = coco_test_data.generate_input(self.workdir, input_details)
    # Move input values from [0, 255] to [-128, 127].
    inputs = inputs - 128
    return [inputs]

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

