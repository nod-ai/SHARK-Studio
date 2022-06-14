# RUN: %PYTHON %s

from shark.shark_importer import SharkImporter
import numpy

model_path = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-64.tflite"

def generate_inputs(input_details):
    args = []
    args.append(
        numpy.random.randint(low=0,
                             high=256,
                             size=input_details[0]["shape"],
                             dtype=input_details[0]["dtype"]))
    return args


if __name__ == '__main__':
    my_shark_importer = SharkImporter(model_path=model_path,
                                      model_type="tflite",
                                      model_source_hub="tfhub",
                                      device="cpu",
                                      dynamic=False,
                                      jit_trace=True)
    # Case1: Use default inputs
    my_shark_importer.compile()
    shark_results = my_shark_importer.forward()
    # Case2: Use manually set inputs
    input_details, output_details = my_shark_importer.get_model_details()
    inputs = generate_inputs(input_details)  # device_inputs
    my_shark_importer.compile(inputs)
    shark_results = my_shark_importer.forward(inputs)
    # print(shark_results)
