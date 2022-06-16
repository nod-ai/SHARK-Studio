# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite"


class DeepLabV3Test(test_util.TFLiteModelTest):

    def __init__(self, *args, **kwargs):
        super(DeepLabV3Test, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(DeepLabV3Test, self).compare_results(iree_results, tflite_results,
                                                   details)
        self.assertTrue(
            numpy.isclose(iree_results[0], tflite_results[0], atol=1e-3).all())

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == '__main__':
    absl.testing.absltest.main()
