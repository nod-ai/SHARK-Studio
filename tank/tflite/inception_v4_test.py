# RUN: %PYTHON %s

import absl.testing
import imagenet_test_data
import numpy
import test_util

# Source https://tfhub.dev/tensorflow/lite-model/inception_v4/1/default/1
model_path = "https://storage.googleapis.com/iree-model-artifacts/inception_v4_299_fp32.tflite"


class InceptionV4Test(test_util.TFLiteModelTest):
    def __init__(self, *args, **kwargs):
        super(InceptionV4Test, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(InceptionV4Test, self).compare_results(iree_results, tflite_results, details)
        self.assertTrue(numpy.isclose(iree_results, tflite_results, atol=1e-4).all())

    def generate_inputs(self, input_details):
        inputs = imagenet_test_data.generate_input(self.workdir, input_details)
        # Normalize inputs to [-1, 1].
        inputs = (inputs.astype("float32") / 127.5) - 1
        return [inputs]

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == "__main__":
    absl.testing.absltest.main()
