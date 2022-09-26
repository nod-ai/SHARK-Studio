import requests
import torch
import torchvision.models as models
from torchvision import transforms
import sys
from shark.shark_inference import SharkInference

MLIR_MODULE = """
func.func @batch_matmul(%arg0: tensor<16x4096x4096xf32>, %arg1: tensor<16x4096x48xf32>)
    -> tensor<16x4096x48xf32>
{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<16x4096x48xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
  %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<16x4096x4096xf32>, tensor<16x4096x48xf32>)
          outs(%1 : tensor<16x4096x48xf32>) -> tensor<16x4096x48xf32>
  return %2 : tensor<16x4096x48xf32>
}
"""

shark_module = SharkInference(MLIR_MODULE, "batch_matmul")
shark_module.compile()
