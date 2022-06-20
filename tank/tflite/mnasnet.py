# RUN: %PYTHON %s
# XFAIL: *
from shark.shark_importer import SharkImporter

model_path = "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.0_224/1/metadata/1?lite-format=tflite"

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
