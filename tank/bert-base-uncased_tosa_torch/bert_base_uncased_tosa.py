from shark.shark_inference import SharkInference
from shark.shark_downloader import download_torch_model

mlir_model, func_name, inputs, golden_out = download_torch_model(
    "bert-base-uncased_tosa"
)

shark_module = SharkInference(
    mlir_model, func_name, device="cpu", mlir_dialect="tosa"
)
shark_module.compile()
result = shark_module.forward(inputs)
print("The obtained result via shark is: ", result)
print("The golden result is:", golden_out)

import numpy as np

result_unsqueeze = np.expand_dims(result, axis=0)

print(
    np.testing.assert_allclose(
        result_unsqueeze, golden_out, rtol=1e-3, atol=1e-3
    )
)
