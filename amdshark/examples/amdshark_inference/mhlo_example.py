from amdshark.amdshark_inference import AMDSharkInference
import numpy as np

mhlo_ir = r"""builtin.module  {
      func.func @forward(%arg0: tensor<1x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<4x4xf32> {
        %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<1x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
        %1 = "mhlo.abs"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
        return %1 : tensor<4x4xf32>
      }
}"""

arg0 = np.ones((1, 4)).astype(np.float32)
arg1 = np.ones((4, 1)).astype(np.float32)

print("Running amdshark on cpu backend")
amdshark_module = AMDSharkInference(mhlo_ir, device="cpu", mlir_dialect="mhlo")

# Generate the random inputs and feed into the graph.
x = amdshark_module.generate_random_inputs()
amdshark_module.compile()
print(amdshark_module.forward(x))

print("Running amdshark on cuda backend")
amdshark_module = AMDSharkInference(mhlo_ir, device="cuda", mlir_dialect="mhlo")
amdshark_module.compile()
print(amdshark_module.forward(x))

print("Running amdshark on vulkan backend")
amdshark_module = AMDSharkInference(mhlo_ir, device="vulkan", mlir_dialect="mhlo")
amdshark_module.compile()
print(amdshark_module.forward(x))
