# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3-dynamic-shapes/int8/predict/1?lite-format=tflite"


# Failure is due to avg_pool2d.
class ImageStylizationTest(test_util.TFLiteModelTest):

    def __init__(self, *args, **kwargs):
        super(ImageStylizationTest, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(ImageStylizationTest,
              self).compare_results(iree_results, tflite_results, details)
        iree = iree_results[0].flatten().astype(numpy.single)
        tflite = tflite_results[0].flatten().astype(numpy.single)
        # Error is not tiny but appears close.
        self.assertTrue(
            numpy.isclose(numpy.max(numpy.abs(iree - tflite)), 0.0,
                          atol=5e-2).all())

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == '__main__':
    absl.testing.absltest.main()
