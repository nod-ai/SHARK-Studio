# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util
import urllib.request

from PIL import Image

model_path = "https://github.com/mlcommons/tiny/raw/0b04bcd402ee28f84e79fa86d8bb8e731d9497b8/v0.5/training/visual_wake_words/trained_models/vww_96_int8.tflite"


# Failure is due to dynamic shapes. This model has a dynamic batch dimension
# and there is not currently supported. Flatbuffer was modified to use static
#  shapes and was otherwise numerically correct.
class VisualWakeWordsTest(test_util.TFLiteModelTest):

    def __init__(self, *args, **kwargs):
        super(VisualWakeWordsTest, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(VisualWakeWordsTest,
              self).compare_results(iree_results, tflite_results, details)
        self.assertTrue(
            numpy.isclose(iree_results[0], tflite_results[0], atol=1).all())

    def generate_inputs(self, input_details):
        img_path = "https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp"
        local_path = "/".join([self.workdir, "person.bmp"])
        urllib.request.urlretrieve(img_path, local_path)

        shape = input_details[0]["shape"]
        input_type = input_details[0]["dtype"]
        im = numpy.array(
            Image.open(local_path).resize(
                (shape[1], shape[2])).convert(mode="RGB"))
        args = [im.reshape(shape).astype(input_type)]
        return args

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == '__main__':
    absl.testing.absltest.main()
