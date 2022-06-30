# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import test_util

model_path = "https://tfhub.dev/tulasiram58827/lite-model/craft-text-detector/dr/1?lite-format=tflite"


# Failure: Resize lowering does not handle inferred dynamic shapes. Furthermore, the entire model
# requires dynamic shape support.
class CraftTextTest(test_util.TFLiteModelTest):
    def __init__(self, *args, **kwargs):
        super(CraftTextTest, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(CraftTextTest, self).compare_results(iree_results, tflite_results, details)

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == "__main__":
    absl.testing.absltest.main()
