from shark.shark_inference import SharkInference
from shark.shark_downloader import download_torch_model


mlir_model, func_name, inputs, golden_out = download_torch_model(
    "microsoft/MiniLM-L12-H384-uncased"
)


shark_module = SharkInference(
    mlir_model, func_name, device="cpu", mlir_dialect="linalg"
)
shark_module.compile()
result = shark_module.forward(inputs)
print("The obtained result via shark is: ", result)
print("The golden result is:", golden_out)


# Let's generate random inputs, currently supported
# for static models.
rand_inputs = shark_module.generate_random_inputs()
rand_results = shark_module.forward(rand_inputs)

print("Running shark_module with random_inputs is: ", rand_results)
