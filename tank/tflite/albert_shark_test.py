# RUN: %PYTHON %s
import numpy as np
from shark.shark_importer import SharkImporter, GenerateInputSharkImporter
import pytest

model_path = "https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1?lite-format=tflite"

# for different model, we have to generate different inputs.
class AlbertInput(GenerateInputSharkImporter):
    def __init__(self, input_details, model_source_hub):
        super(AlbertInput, self).__init__(input_details, model_source_hub)

    # Inputs modified to be useful albert inputs.
    def generate_inputs(self):
        for input in self.input_details:
            print("\t%s, %s", str(input["shape"]), input["dtype"].__name__)

        args = []
        args.append(
            np.random.randint(low=0, high=256, size=self.input_details[0]["shape"], dtype=self.input_details[0]["dtype"]))
        args.append(np.ones(shape=self.input_details[1]["shape"], dtype=self.input_details[1]["dtype"]))
        args.append(np.zeros(shape=self.input_details[2]["shape"], dtype=self.input_details[2]["dtype"]))
        return args

# A specific case can be run by commenting different cases. Runs all the test
# across cpu, gpu and vulkan according to available drivers.
pytest_param = pytest.mark.parametrize(
    ('dynamic', 'device'),
    [
        pytest.param(False, 'cpu'),
        # TODO: Language models are failing for dynamic case..
        pytest.param(True, 'cpu', marks=pytest.mark.skip),
    ])


@pytest_param
def test_albert(dynamic, device):
    my_shark_importer = SharkImporter(model_path=model_path,
                                      model_type="tflite",
                                      model_source_hub="tfhub",
                                      device=device,
                                      dynamic=dynamic,
                                      jit_trace=True
                                      )
    input_details, output_details = my_shark_importer.get_model_details()
    albert_inputs_obj = AlbertInput(input_details, "tfhub")
    inputs = albert_inputs_obj.generate_inputs() # device_inputs
    my_shark_importer.setup_inputs(inputs)
    shark_results = my_shark_importer.compile_and_execute()
    # print(shark_results)

if __name__ == '__main__':
    test_albert(False, "cpu")
