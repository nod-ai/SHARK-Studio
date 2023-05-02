from transformers import AutoTokenizer, FlaxAutoModel
import torch
import jax
from typing import Union, Dict, List
import numpy as np
from shark.shark_inference import SharkInference
import io

NumpyTree = Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]]


def convert_torch_tensor_tree_to_numpy(
    tree: Union[torch.tensor, Dict[str, torch.tensor], List[torch.tensor]]
) -> NumpyTree:
    return jax.tree_util.tree_map(
        lambda torch_tensor: torch_tensor.cpu().detach().numpy(), tree
    )


def convert_int64_to_int32(tree: NumpyTree) -> NumpyTree:
    return jax.tree_util.tree_map(
        lambda tensor: np.array(tensor, dtype=np.int32)
        if tensor.dtype == np.int64
        else tensor,
        tree,
    )


def get_sample_input():
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/MiniLM-L12-H384-uncased"
    )
    inputs_torch = tokenizer("Hello, World!", return_tensors="pt")
    return convert_int64_to_int32(
        convert_torch_tensor_tree_to_numpy(inputs_torch.data)
    )


def export_to_mlir(sample_input: NumpyTree):
    model = FlaxAutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    model_mlir = jax.jit(model).lower(**sample_input).compiler_ir()
    byte_stream = io.BytesIO()
    model_mlir.operation.write_bytecode(file=byte_stream)
    return byte_stream.getvalue()


sample_input = get_sample_input()
mlir = export_to_mlir(sample_input)

# Compile and load module.
shark_inference = SharkInference(
    mlir_module=mlir, mlir_dialect="mhlo", device="cpu"
)
shark_inference.compile()

# Run main function.
print(shark_inference("main", jax.tree_util.tree_flatten(sample_input)[0]))
