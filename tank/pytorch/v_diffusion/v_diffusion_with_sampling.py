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

sys.path.append("v-diffusion-pytorch")

from CLIP import clip
from diffusion import get_model, sampling, utils
import torch_mlir

# Load the models
model = get_model("cc12m_1_cfg")()
_, side_y, side_x = model.shape
checkpoint = "v-diffusion-pytorch/checkpoints/cc12m_1_cfg.pth"
model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
model = model.eval().requires_grad_(False)
clip_model = clip.load(model.clip_model, jit=True, device="cpu")[0]

prompt = "New York City, oil on canvas"

weight = 1
n_images = 1
steps = 2

target_embed = clip_model.encode_text(clip.tokenize(prompt))
x = torch.randn([n_images, 3, side_y, side_x], device="cpu")
t = torch.linspace(1, 0, steps + 1, device="cpu")[:-1]


def model_inference(x, t, target_embed):
    ddpm_crossover = 0.48536712
    cosine_crossover = 0.80074257
    big_t = t * (1 + cosine_crossover - ddpm_crossover)
    ddpm_t = big_t + ddpm_crossover - cosine_crossover
    log_snr = -torch.special.expm1(1e-4 + 10 * ddpm_t**2).log()
    alpha, sigma = log_snr.sigmoid().sqrt(), log_snr.neg().sigmoid().sqrt()
    ddpm_part = torch.atan2(sigma, alpha) / math.pi * 2
    steps_ = torch.where(big_t < cosine_crossover, big_t, ddpm_part)
    ts = x.new_ones([1])
    steps_1 = torch.cat([steps_, steps_.new_zeros([1])])
    t_1, t_2 = steps_1[0] * ts, steps_1[1] * ts
    t_mid = (t_2 + t_1) / 2
    alphas, sigmas = torch.cos(t_1 * math.pi / 2), torch.sin(t_1 * math.pi / 2)
    x_in = x.repeat([2, 1, 1, 1])
    t_in = t_1.repeat([2])
    clip_embed_repeat = target_embed.repeat([1, 1])
    clip_embed_in = torch.cat(
        [torch.zeros_like(clip_embed_repeat), clip_embed_repeat]
    )
    v_uncond, v_cond = model(x_in, t_in, clip_embed_in).chunk(2, dim=0)
    v = v_uncond + (v_cond - v_uncond) * 1
    eps_1 = x * sigmas[:, None, None, None] + v * alphas[:, None, None, None]
    next_alphas, next_sigmas = (
        torch.cos(t_mid * math.pi / 2),
        torch.sin(t_mid * math.pi / 2),
    )
    pred = (x - eps_1 * sigmas[:, None, None, None]) / alphas[
        :, None, None, None
    ]
    x_1 = (
        pred * next_alphas[:, None, None, None]
        + eps_1 * next_sigmas[:, None, None, None]
    )
    x_in_1 = x_1.repeat([2, 1, 1, 1])
    t_in_1 = t_mid.repeat([2])
    # The call `model` is resulting in an error.
    v_uncond_1, v_cond_1 = model(x_in_1, t_in_1, clip_embed_in).chunk(2, dim=0)
    v_1 = v_uncond_1 + (v_cond_1 - v_uncond_1) * 1
    eps_2 = (
        x_1 * next_sigmas[:, None, None, None]
        + v_1 * next_alphas[:, None, None, None]
    )
    eps_prime = (eps_1 + 2 * eps_2) / 3  # + 2 * eps_3 + eps_4) / 6
    next_alphas_1, next_sigmas_1 = (
        torch.cos(t_2 * math.pi / 2),
        torch.sin(t_2 * math.pi / 2),
    )
    pred_1 = (x - eps_prime * sigmas[:, None, None, None]) / alphas[
        :, None, None, None
    ]
    x_new = (
        pred_1 * next_alphas_1[:, None, None, None]
        + eps_prime * next_sigmas_1[:, None, None, None]
    )
    return x_new


# model_inference(x, t, target_embed)

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
            torch.ops.aten.split_with_sizes,
        ]
    ),
)(x, t, target_embed)

# print(fx_g.graph)

fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
fx_g.recompile()


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


strip_overloads(fx_g)

ts_g = torch.jit.script(fx_g)
# ts_g = torch.jit.trace(fx_g, (x, t, target_embed)).eval()
# tmp = torch.jit.freeze(ts_g)
# print (tmp.graph)

# temp = tempfile.NamedTemporaryFile(
#     suffix="_shark_ts", prefix="temp_ts_"
# )
# ts_g.save(temp.name)
# new_ts = torch.jit.load(temp.name)
# print (ts_g.graph)

module = torch_mlir.compile(
    ts_g,
    [x, t, target_embed],
    torch_mlir.OutputType.RAW,
    use_tracing=False,
    verbose=True,
)
# print(module)
module.dump()
