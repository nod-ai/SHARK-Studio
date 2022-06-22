# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util
import urllib.request

from PIL import Image

model_path = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"


# Currently failing further in the linalg stack:
#   Invalid cast from ui8 to f32 TODO: make tfl.cast insert a rescale for ui8
class LightningI8Test(test_util.TFLiteModelTest):
    def __init__(self, *args, **kwargs):
        super(LightningI8Test, self).__init__(model_path, *args, **kwargs)

    # Optional utility for visualizing results on the input image. This verifies
    # the model works-ish as the dots are in roughly the same area. Still need to
    # debug the source of numerical differences.
    def plot_results(self, iree_results, tflite_results, details):
        from matplotlib import pyplot as plt

        local_path = "/".join([self.workdir, "person.jpg"])
        im = numpy.array(Image.open(local_path))

        width = im.shape[0]
        height = im.shape[1]

        tflite_result = tflite_results[0]
        iree_result = iree_results[0]

        tflite_x = tflite_result[0, 0, :, 0] * width
        tflite_y = tflite_result[0, 0, :, 1] * height

        iree_x = iree_result[0, 0, :, 0] * width
        iree_y = iree_result[0, 0, :, 1] * height

        plt.imshow(im)
        plt.scatter(tflite_y, tflite_x, label="tflite")
        plt.scatter(iree_y, iree_x, label="iree")
        plt.legend()
        plt.show()

    def compare_results(self, iree_results, tflite_results, details):
        super(LightningI8Test, self).compare_results(
            iree_results, tflite_results, details
        )
        # This value is a discretized location of the persons joints. If we are
        # *close* to the expected position we can consider this good enough.
        self.assertTrue(
            numpy.isclose(
                iree_results[0][:, :, :, 0],
                tflite_results[0][:, :, :, 0],
                atol=25e-3,
            ).all()
        )
        self.assertTrue(
            numpy.isclose(
                iree_results[0][:, :, :, 1],
                tflite_results[0][:, :, :, 1],
                atol=25e-3,
            ).all()
        )
        # self.plot_results(iree_results, tflite_results, details)

    def generate_inputs(self, input_details):
        img_path = "https://github.com/tensorflow/examples/raw/master/lite/examples/pose_estimation/raspberry_pi/test_data/image3.jpeg"
        local_path = "/".join([self.workdir, "person.jpg"])
        urllib.request.urlretrieve(img_path, local_path)

        shape = input_details[0]["shape"]
        im = numpy.array(Image.open(local_path).resize((shape[1], shape[2])))
        args = [im.reshape(shape)]
        return args

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == "__main__":
    absl.testing.absltest.main()
