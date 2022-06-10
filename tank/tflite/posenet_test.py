# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util

model_path = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite"


class PoseTest(test_util.TFLiteModelTest):

    def __init__(self, *args, **kwargs):
        super(PoseTest, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(PoseTest, self).compare_results(iree_results, tflite_results,
                                              details)
        self.assertTrue(
            numpy.isclose(iree_results[0], tflite_results[0], atol=1e-3).all())
        self.assertTrue(
            numpy.isclose(iree_results[1], tflite_results[1], atol=1e-2).all())
        self.assertTrue(
            numpy.isclose(iree_results[2], tflite_results[2], atol=1e-2).all())
        self.assertTrue(
            numpy.isclose(iree_results[3], tflite_results[3], atol=1e-3).all())

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == '__main__':
    absl.testing.absltest.main()
