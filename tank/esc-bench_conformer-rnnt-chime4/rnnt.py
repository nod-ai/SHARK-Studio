from shark.shark_inference import SharkInference
from shark.shark_downloader import download_torch_model

mlir_model, func_name, inputs, golden_out = download_torch_model(
    "esc-bench/conformer-rnnt-chime4"
)

shark_module = SharkInference(
    mlir_model, func_name, device="cpu", mlir_dialect="linalg"
)
shark_module.compile()
result = shark_module.forward(inputs)