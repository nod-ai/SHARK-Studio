"""classifier-free guidance sampling from a diffusion model."""

from functools import partial
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange

from shark.shark_inference import SharkInference
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import torch_mlir

import sys

sys.path.append("models/diffusion/v-diffusion-pytorch")

from CLIP import clip
from diffusion import get_model, get_models, sampling, utils

import gradio as gr

MODULE_DIR = Path(__file__).resolve().parent

set_global_parameters = False
device = None
model = None
checkpoint = None
clip_model = None


def parse_prompt(prompt, default_weight=3.0):
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", default_weight][len(vals) :]
    print(vals[1])
    print(vals[0])
    return vals[0], float(vals[1])


def run(x, steps, shark_module, args):
    def compiled_cfg_model_fn(x, t):
        x_ny = x.detach().numpy()
        t_ny = t.detach().numpy()
        inputs = (x_ny, t_ny)
        result = shark_module.forward(inputs)
        return torch.from_numpy(result)

    return sampling.plms_sample(compiled_cfg_model_fn, x, steps, {})


def run_all(
    x,
    t,
    steps,
    n,
    batch_size,
    side_x,
    side_y,
    shark_module,
    args,
):
    x = torch.randn([n, 3, side_y, side_x], device=device)
    t = torch.linspace(1, 0, args["steps"] + 1, device=device)[:-1]
    steps = utils.get_spliced_ddpm_cosine_schedule(t)
    pil_images = []
    for i in trange(0, n, batch_size):
        cur_batch_size = min(n - i, batch_size)
        outs = run(x[i : i + cur_batch_size], steps, shark_module, args)
        for j, out in enumerate(outs):
            pil_images.append(utils.to_pil_image(out))
    return pil_images[0]


def cache_model():
    global set_global_parameters
    global device
    global model
    global checkpoint
    global clip_model
    if not set_global_parameters:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model("cc12m_1_cfg")()
        checkpoint = MODULE_DIR / f"checkpoints/cc12m_1_cfg.pth"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        if device.type == "cuda":
            model = model.half()
        model = model.to(device).eval().requires_grad_(False)
        clip_model_name = (
            model.clip_model if hasattr(model, "clip_model") else "ViT-B/16"
        )
        clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
        clip_model.eval().requires_grad_(False)
        set_global_parameters = True


def vdiff_inf(prompts: str, n, bs, steps, _device):
    global device
    global model
    global checkpoint
    global clip_model

    args = {}
    target_embeds = []
    weights = []
    args["prompts"] = prompts
    args["batch_size"] = int(bs)
    args["n"] = int(n)
    args["seed"] = 0
    args["steps"] = int(steps)
    args["device"] = _device

    cache_model()

    _, side_y, side_x = model.shape
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    zero_embed = torch.zeros([1, clip_model.visual.output_dim], device=device)
    target_embeds.append(zero_embed)

    prompt_list = args["prompts"].rsplit(";")
    for prompt in prompt_list:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(
            clip_model.encode_text(clip.tokenize(txt).to(device)).float()
        )
        weights.append(weight)
    weights = torch.tensor([1 - sum(weights), *weights], device=device)

    torch.manual_seed(args["seed"])

    x = torch.randn([args["n"], 3, side_y, side_x], device=device)
    t = torch.linspace(1, 0, args["steps"] + 1, device=device)[:-1]
    steps = utils.get_spliced_ddpm_cosine_schedule(t)
    min_batch_size = min(args["n"], args["batch_size"])
    x_in = x[0:min_batch_size, :, :, :]
    ts = x_in.new_ones([x_in.shape[0]])
    t_in = t[0] * ts

    def cfg_model_fn(x, t):
        n = x.shape[0]
        n_conds = len(target_embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = torch.cat([*target_embeds]).repeat([n, 1])
        vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
        v = vs.mul(weights[:, None, None, None, None]).sum(0)
        return v

    fx_g = make_fx(
        cfg_model_fn,
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
    )(x_in, t_in)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    for node in fx_g.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    fx_g.recompile()

    ts_g = torch.jit.script(fx_g)

    module = torch_mlir.compile(
        ts_g,
        [x_in, t_in],
        torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,
    )

    mlir_model = module
    func_name = "forward"
    shark_module = SharkInference(
        mlir_model, func_name, device=args["device"], mlir_dialect="linalg"
    )
    shark_module.compile()
    return (
        run_all(
            x,
            t,
            args["steps"],
            args["n"],
            args["batch_size"],
            side_x,
            side_y,
            shark_module,
            args,
        ),
        "Testing..",
    )
