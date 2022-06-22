# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/google/lite-model/spice/1?lite-format=tflite"


# Currently unsupported:
# 1. Multiple unsupported dynamic operations (tfl.stride, range, gather).
# 2. Static version blocked by tfl.range not having a lowering for static fixed shapes.
class SpiceTest(test_util.TFLiteModelTest):
    def __init__(self, *args, **kwargs):
        super(SpiceTest, self).__init__(model_path, *args, **kwargs)

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == "__main__":
    absl.testing.absltest.main()
