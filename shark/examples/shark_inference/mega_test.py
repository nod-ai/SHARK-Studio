import torch
import torch_mlir
from shark.shark_inference import SharkInference
from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    args,
)
from MEGABYTE_pytorch import MEGABYTE

import os


class MegaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MEGABYTE(
            num_tokens=16000,  # number of tokens
            dim=(
                512,
                256,
            ),  # transformer model dimension (512 for coarsest, 256 for fine in this example)
            max_seq_len=(
                1024,
                4,
            ),  # sequence length for global and then local. this can be more than 2
            depth=(
                6,
                4,
            ),  # number of layers for global and then local. this can be more than 2, but length must match the max_seq_len's
            dim_head=64,  # dimension per head
            heads=8,  # number of attention heads
            flash_attn=True,  # use flash attention
        )

    def forward(self, input):
        return self.model(input)


megaModel = MegaModel()
input = [torch.randint(0, 16000, (1, 1024, 4))]

# CURRENTLY IT BAILS OUT HERE BECAUSE OF MISSING OP LOWERINGS :-
# 1. aten.alias
shark_module, _ = compile_through_fx(
    megaModel,
    inputs=input,
    extended_model_name="mega_shark",
    debug=False,
    generate_vmfb=True,
    save_dir=os.getcwd(),
    extra_args=[],
    base_model_id=None,
    model_name="mega_shark",
    precision=None,
    return_mlir=True,
    device="cuda",
)
# logits = model(x)


def print_output_info(output, msg):
    print("\n", msg)
    print("\n\t", output.shape)


ans = shark_module("forward", input)
print_output_info(torch.from_numpy(ans), "SHARK's output")

ans = megaModel.forward(*input)
print_output_info(ans, "ORIGINAL Model's output")

# and sample from the logits accordingly
# or you can use the generate function

# NEED TO LOOK AT THIS LATER IF REQUIRED IN SHARK.
# sampled = model.generate(temperature = 0.9, filter_thres = 0.9) # (1, 1024, 4)
