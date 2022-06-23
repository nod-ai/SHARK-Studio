from shark.shark_inference import SharkInference
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

print("Running shark on cpu backend")
shark_module = SharkInference(
    mhlo_ir, function_name="forward", device="cpu", mlir_dialect="mhlo"
)
shark_module.compile()
print(shark_module.forward((arg0, arg1)))

print("Running shark on cuda backend")
shark_module = SharkInference(
    mhlo_ir, function_name="forward", device="cuda", mlir_dialect="mhlo"
)
shark_module.compile()
print(shark_module.forward((arg0, arg1)))

print("Running shark on vulkan backend")
shark_module = SharkInference(
    mhlo_ir, function_name="forward", device="vulkan", mlir_dialect="mhlo"
)
shark_module.compile()
print(shark_module.forward((arg0, arg1)))
