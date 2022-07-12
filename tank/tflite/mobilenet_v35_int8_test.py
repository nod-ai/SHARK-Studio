# RUN: %PYTHON %s

import absl.testing
import imagenet_test_data
import numpy
import test_util

model_path = "https://storage.googleapis.com/tf_model_garden/vision/mobilenet/v3.5multiavg_1.0_int8/mobilenet_v3.5multiavg_1.00_224_int8.tflite"


class MobilenetV35Int8Test(test_util.TFLiteModelTest):
    def __init__(self, *args, **kwargs):
        super(MobilenetV35Int8Test, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(MobilenetV35Int8Test, self).compare_results(
            iree_results, tflite_results, details
        )
        # The difference here is quite high for a dequantized output.
        self.assertTrue(
            numpy.isclose(iree_results, tflite_results, atol=0.5).all()
        )

        # Make sure the predicted class is the same.
        iree_predicted_class = numpy.argmax(iree_results[0][0])
        tflite_predicted_class = numpy.argmax(tflite_results[0][0])
        self.assertEqual(iree_predicted_class, tflite_predicted_class)

    def generate_inputs(self, input_details):
        inputs = imagenet_test_data.generate_input(self.workdir, input_details)
        # Normalize inputs to [-1, 1].
        inputs = (inputs.astype("float32") / 127.5) - 1
        return [inputs]

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == "__main__":
    absl.testing.absltest.main()
