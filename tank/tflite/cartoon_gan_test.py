# RUN: %PYTHON %s
# REQUIRES: hugetest

import absl.testing
import test_util

model_path = (
    "https://tfhub.dev/sayakpaul/lite-model/cartoongan/dr/1?lite-format=tflite"
)


class CartoonGanTest(test_util.TFLiteModelTest):
    def __init__(self, *args, **kwargs):
        super(CartoonGanTest, self).__init__(model_path, *args, **kwargs)

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == "__main__":
    absl.testing.absltest.main()
