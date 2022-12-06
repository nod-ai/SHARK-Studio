from shark.shark_inference import SharkInference
from shark.shark_downloader import download_torch_model


mlir_model, func_name, inputs, golden_out = download_torch_model("v_diffusion")

shark_module = SharkInference(
    mlir_model, func_name, device="vulkan", mlir_dialect="linalg"
)
shark_module.compile()
result = shark_module.forward(inputs)
print("The obtained result via shark is: ", result)
print("The golden result is:", golden_out)
