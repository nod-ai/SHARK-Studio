# RUN: %PYTHON %s
import numpy as np
from shark.shark_importer import SharkImporter, GenerateInputSharkImporter
import pytest
from shark.iree_utils import check_device_drivers

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

    def compare_results(self, iree_results, tflite_results, details):
        print("len(iree_results):", len(iree_results))
        print("len(tflite_results):", len(tflite_results))
        assert (len(iree_results) == len(tflite_results), f"Number of results do not match")

        for i in range(len(details)):
            iree_result = iree_results[i]
            tflite_result = tflite_results[i]
            iree_result = iree_result.astype(np.single)
            tflite_result = tflite_result.astype(np.single)
            assert (iree_result.shape == tflite_result.shape)
            maxError = np.max(np.abs(iree_result - tflite_result))
            print("Max error (%d): %f", i, maxError)

        assert(np.isclose(iree_results[0], tflite_results[0], atol=1e-4).all() == True)
        assert(np.isclose(iree_results[1], tflite_results[1], atol=1e-4).all() == True)

# A specific case can be run by commenting different cases. Runs all the test
# across cpu, gpu and vulkan according to available drivers.
pytest_param = pytest.mark.parametrize(
    ('dynamic', 'device'),
    [
        pytest.param(False, 'cpu'),
        # TODO: Language models are failing for dynamic case..
        pytest.param(True, 'cpu', marks=pytest.mark.skip),
        pytest.param(False,
                     'gpu',
                     marks=pytest.mark.skipif(check_device_drivers("gpu"),
                                              reason="nvidia-smi not found")),
        pytest.param(True,
                     'gpu',
                     marks=pytest.mark.skipif(check_device_drivers("gpu"),
                                              reason="nvidia-smi not found")),
        pytest.param(
            False,
            'vulkan',
            marks=pytest.mark.skipif(
                check_device_drivers("vulkan"),
                reason=
                "vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )),
        pytest.param(
            True,
            'vulkan',
            marks=pytest.mark.skipif(
                check_device_drivers("vulkan"),
                reason=
                "vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )),
    ])


@pytest_param
def test_albert(dynamic, device):
    my_shark_importer = SharkImporter(model_path=model_path,
                                      model_type="tflite",
                                      model_source_hub="tfhub",
                                      exe_config='cpu',
                                      device=device,
                                      dynamic=dynamic,
                                      jit_trace=True
                                      )
    input_details, output_details = my_shark_importer.setup_tflite()
    albert_inputs_obj = AlbertInput(input_details, "tfhub")
    inputs = albert_inputs_obj.generate_inputs() # device_inputs
    my_shark_importer.setup_input(inputs)
    iree_results, tflite_results = my_shark_importer.compile_and_execute()
    albert_inputs_obj.compare_results(iree_results, tflite_results, output_details)

if __name__ == '__main__':
    test_albert(False, "cpu")