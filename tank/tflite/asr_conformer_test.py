# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import test_util

model_path = "https://tfhub.dev/neso613/lite-model/ASR_TFLite/pre_trained_models/English/1?lite-format=tflite"


# Failure is due to dynamic shapes:
# - Some improvements to tfl.strided_slice lowering are next steps
class AsrConformerTest(test_util.TFLiteModelTest):

    def __init__(self, *args, **kwargs):
        super(AsrConformerTest, self).__init__(model_path, *args, **kwargs)

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == '__main__':
    absl.testing.absltest.main()
