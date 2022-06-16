# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import test_util

model_path = "https://tfhub.dev/tulasiram58827/lite-model/keras-ocr/dr/2?lite-format=tflite"


class KerasOCRTest(test_util.TFLiteModelTest):

    def __init__(self, *args, **kwargs):
        super(KerasOCRTest, self).__init__(model_path, *args, **kwargs)

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == '__main__':
    absl.testing.absltest.main()
