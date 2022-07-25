# # Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# # See https://llvm.org/LICENSE.txt for license information.
# # SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# # Also available under a BSD-style license. See LICENSE.

import torch

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import tempfile

import math
import sys
import gc

from torchvision import utils as tv_utils
from torchvision.transforms import functional as TF
from tqdm.notebook import trange, tqdm

sys.path.append("v-diffusion-pytorch")

import clip
from diffusion import get_model, sampling, utils
import torch_mlir


# Load the models
model = get_model("cc12m_1_cfg")()
_, side_y, side_x = model.shape
model = model.eval().requires_grad_(False)
clip_model = clip.load(model.clip_model, jit=True, device="cpu")[0]

prompt = "New York City, oil on canvas"

weight = 1
n_images = 1
steps = 2

target_embed = clip_model.encode_text(clip.tokenize(prompt))
x = torch.randn([n_images, 3, side_y, side_x], device="cpu")
t = torch.linspace(1, 0, steps + 1, device="cpu")[:-1]

n = x.shape[0]
x_in = x.repeat([2, 1, 1, 1])
t_in = t
clip_embed_repeat = target_embed.repeat([n, 1])
clip_embed_in = torch.cat(
    [torch.zeros_like(clip_embed_repeat), clip_embed_repeat]
)


def model_inference(x_in, t_in, clip_embed_in):
    return model(x_in, t_in, clip_embed_in)


fx_g = make_fx(
    model_inference,
    decomposition_table=get_decompositions(
        [
            torch.ops.aten.embedding_dense_backward,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.slice_backward,
            torch.ops.aten.select_backward,
            torch.ops.aten.norm.ScalarOpt_dim,
            torch.ops.aten.native_group_norm,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.split.Tensor,
        ]
    ),
)(x_in, t_in, clip_embed_in)

fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
fx_g.recompile()

ts_g = torch.jit.trace(fx_g, (x_in, t_in, clip_embed_in))
temp = tempfile.NamedTemporaryFile(suffix="_shark_ts", prefix="temp_ts_")
ts_g.save(temp.name)
new_ts = torch.jit.load(temp.name)

module = torch_mlir.compile(
    new_ts,
    [x_in, t_in, clip_embed_in],
    torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=False,
)
module.dump()
