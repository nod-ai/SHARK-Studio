from amdshark.amdshark_inference import AMDSharkInference
from amdshark.amdshark_downloader import download_model


mlir_model, func_name, inputs, golden_out = download_model(
    "v_diffusion", frontend="torch"
)

amdshark_module = AMDSharkInference(
    mlir_model, device="vulkan", mlir_dialect="linalg"
)
amdshark_module.compile()
result = amdshark_module.forward(inputs)
print("The obtained result via amdshark is: ", result)
print("The golden result is:", golden_out)
