from amdshark.amdshark_inference import AMDSharkInference
from amdshark.amdshark_downloader import download_model


mlir_model, func_name, inputs, golden_out = download_model(
    "microsoft/MiniLM-L12-H384-uncased",
    frontend="torch",
)


amdshark_module = AMDSharkInference(mlir_model, device="cpu", mlir_dialect="linalg")
amdshark_module.compile()
result = amdshark_module.forward(inputs)
print("The obtained result via amdshark is: ", result)
print("The golden result is:", golden_out)


# Let's generate random inputs, currently supported
# for static models.
rand_inputs = amdshark_module.generate_random_inputs()
rand_results = amdshark_module.forward(rand_inputs)

print("Running amdshark_module with random_inputs is: ", rand_results)
