# # Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# # See https://llvm.org/LICENSE.txt for license information.
# # SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# # Also available under a BSD-style license. See LICENSE.

import torch

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch._decomp import register_decomposition
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


@register_decomposition(torch.ops.aten.upsample_bilinear2d.vec)
def upsample_bilinear2d_vec(input, output_size, align_corners, scale_factors):
    # get dimensions of original image
    n_batch, n_channels, in_h, in_w = input.shape

    if output_size is not None:
        out_h = float(output_size[0])
        out_w = float(output_size[1])
    elif scale_factors is not None:
        out_h = in_h * scale_factors[0]
        out_w = in_w * scale_factors[1]

    # Calculate horizontal and vertical scaling factor
    if out_h > 1:
        if align_corners:
            h_scale_factor = (in_h - 1) / (int(out_h) - 1)
        else:
            h_scale_factor = in_h / out_h
    else:
        h_scale_factor = 0.0

    if out_w > 1:
        if align_corners:
            w_scale_factor = (in_w - 1) / (int(out_w) - 1)
        else:
            w_scale_factor = in_w / out_w
    else:
        w_scale_factor = 0.0

    i = torch.arange(out_h, dtype=input.dtype, device=input.device)
    j = torch.arange(out_w, dtype=input.dtype, device=input.device)

    if align_corners:
        x = h_scale_factor * i
        y = w_scale_factor * j
    else:
        x = (h_scale_factor * (i + 0.5) - 0.5).clamp(min=0.0)
        y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)

    x_floor = torch.floor(x)
    x_ceil = torch.minimum(torch.ceil(x), torch.tensor(in_h - 1))
    y_floor = torch.floor(y)
    y_ceil = torch.minimum(torch.ceil(y), torch.tensor(in_w - 1))

    x_view = x.view(1, 1, len(x), 1)
    x_floor_view = x_floor.view(1, 1, len(x_floor), 1)
    x_ceil_view = x_ceil.view(1, 1, len(x_ceil), 1)

    y_view = y.view(1, 1, 1, len(y))
    y_floor_view = y_floor.view(1, 1, 1, len(y_floor))
    y_ceil_view = y_ceil.view(1, 1, 1, len(y_ceil))

    v1 = input[:, :, x_floor.to(torch.int64), :][
        :, :, :, y_floor.to(torch.int64)
    ]
    v2 = input[:, :, x_ceil.to(torch.int64), :][
        :, :, :, y_floor.to(torch.int64)
    ]
    v3 = input[:, :, x_floor.to(torch.int64), :][
        :, :, :, y_ceil.to(torch.int64)
    ]
    v4 = input[:, :, x_ceil.to(torch.int64), :][
        :, :, :, y_ceil.to(torch.int64)
    ]
    q1 = torch.mul(v1, x_ceil_view - x_view) + torch.mul(
        v2, x_view - x_floor_view
    )
    q2 = torch.mul(v3, x_ceil_view - x_view) + torch.mul(
        v4, x_view - x_floor_view
    )
    result = torch.mul(q1, y_ceil_view - y_view) + torch.mul(
        q2, y_view - y_floor_view
    )

    # When (x_ceil == x_floor) and (y_ceil == y_floor).
    result_cond1 = input[:, :, x.to(torch.int64), :][
        :, :, :, y.to(torch.int64)
    ]

    # When (x_ceil == x_floor).
    q1 = input[:, :, x.to(torch.int64), :][:, :, :, y_floor.to(torch.int64)]
    q2 = input[:, :, x.to(torch.int64), :][:, :, :, y_ceil.to(torch.int64)]
    result_cond2 = torch.mul(q1, y_ceil_view - y_view) + torch.mul(
        q2, y_view - y_floor_view
    )

    # When (y_ceil == y_floor).
    q1 = input[:, :, x_floor.to(torch.int64), :][:, :, :, y.to(torch.int64)]
    q2 = input[:, :, x_ceil.to(torch.int64), :][:, :, :, y.to(torch.int64)]
    result_cond3 = torch.mul(q1, x_ceil_view - x_view) + torch.mul(
        q2, x_view - x_floor_view
    )

    result = torch.where(
        torch.eq(x_ceil_view, x_floor_view), result_cond2, result
    )
    result = torch.where(
        torch.eq(y_ceil_view, y_floor_view), result_cond3, result
    )
    result = torch.where(
        torch.logical_and(
            torch.eq(x_ceil_view, x_floor_view),
            torch.eq(y_ceil_view, y_floor_view),
        ),
        result_cond1,
        result,
    )

    return result


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
