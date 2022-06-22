# RUN: %PYTHON %s
import numpy as np
from shark.shark_importer import SharkImporter
import pytest

model_path = "https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1?lite-format=tflite"


# Inputs modified to be useful albert inputs.
def generate_inputs(input_details):
    for input in input_details:
        print("\t%s, %s", str(input["shape"]), input["dtype"].__name__)

    args = []
    args.append(
        np.random.randint(
            low=0,
            high=256,
            size=input_details[0]["shape"],
            dtype=input_details[0]["dtype"],
        )
    )
    args.append(
        np.ones(
            shape=input_details[1]["shape"], dtype=input_details[1]["dtype"]
        )
    )
    args.append(
        np.zeros(
            shape=input_details[2]["shape"], dtype=input_details[2]["dtype"]
        )
    )
    return args


# A specific case can be run by commenting different cases. Runs all the test
# across cpu, gpu and vulkan according to available drivers.
pytest_param = pytest.mark.parametrize(
    ("dynamic", "device"),
    [
        pytest.param(False, "cpu"),
        # TODO: Language models are failing for dynamic case..
        pytest.param(True, "cpu", marks=pytest.mark.skip),
    ],
)


@pytest_param
def test_albert(dynamic, device):
    my_shark_importer = SharkImporter(
        model_path=model_path,
        model_type="tflite",
        model_source_hub="tfhub",
        device=device,
        dynamic=dynamic,
        jit_trace=True,
    )
    input_details, output_details = my_shark_importer.get_model_details()
    inputs = generate_inputs(input_details)  # device_inputs
    my_shark_importer.compile(inputs)
    shark_results = my_shark_importer.forward(inputs)
    # print(shark_results)
