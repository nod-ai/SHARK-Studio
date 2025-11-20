from amdshark.amdshark_inference import AMDSharkInference
from amdshark.amdshark_downloader import download_model

mlir_model, func_name, inputs, golden_out = download_model(
    "bert-base-uncased_tosa",
    frontend="torch",
)

amdshark_module = AMDSharkInference(
    mlir_model, func_name, device="cpu", mlir_dialect="tosa"
)
amdshark_module.compile()
result = amdshark_module.forward(inputs)
print("The obtained result via amdshark is: ", result)
print("The golden result is:", golden_out)

import numpy as np

result_unsqueeze = np.expand_dims(result, axis=0)

print(
    np.testing.assert_allclose(
        result_unsqueeze, golden_out, rtol=1e-3, atol=1e-3
    )
)
