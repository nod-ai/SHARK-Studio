import argparse
import os
from functools import partial

import clip
import torch
from torchvision import transforms
from tqdm import trange

try:
    from diffusion import get_model, sampling, utils
except ModuleNotFoundError:
    print("You need to download v-diffusion source from https://github.com/crowsonkb/v-diffusion-pytorch")
    raise

torch.manual_seed(0)


def parse_prompt(prompt, default_weight=3.0):
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", default_weight][len(vals) :]
    return vals[0], float(vals[1])


args = argparse.Namespace(
    prompts=["New York City, oil on canvas"],
    batch_size=1,
    device="cuda",
    model="cc12m_1_cfg",
    n=1,
    steps=10,
)

device = torch.device(args.device)
print("Using device:", device)

model = get_model(args.model)()
_, side_y, side_x = model.shape
checkpoint = f"{args.model}.pth"
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

model = model.to(device).eval().requires_grad_(False)
clip_model_name = model.clip_model if hasattr(model, "clip_model") else "ViT-B/16"
clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
clip_model.eval().requires_grad_(False)
normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)

zero_embed = torch.zeros([1, clip_model.visual.output_dim], device=device)
target_embeds, weights = [zero_embed], []

txt, weight = parse_prompt(args.prompts[0])
target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
weights.append(weight)

weights = torch.tensor([1 - sum(weights), *weights], device=device)


def cfg_model_fn(model, x, t):
    n = x.shape[0]
    n_conds = len(target_embeds)
    x_in = x.repeat([n_conds, 1, 1, 1])
    t_in = t.repeat([n_conds])
    clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
    vs = model(x_in, t_in, clip_embed_in).view([n_conds, n, *x.shape[1:]])
    v = vs.mul(weights[:, None, None, None, None]).sum(0)
    return v


x = torch.randn([args.n, 3, side_y, side_x], device=device)
t = torch.linspace(1, 0, args.steps + 1, device=device)[:-1]


def repro(model):
    if device.type == "cuda":
        model = model.half()

    steps = utils.get_spliced_ddpm_cosine_schedule(t)
    for i in trange(0, args.n, args.batch_size):
        cur_batch_size = min(args.n - i, args.batch_size)
        outs = sampling.plms_sample(partial(cfg_model_fn, model), x[i : i + cur_batch_size], steps, {})
        for j, out in enumerate(outs):
            utils.to_pil_image(out).save(f"out_{i + j:05}.png")


def trace(model, x, t):
    n = x.shape[0]
    n_conds = len(target_embeds)
    x_in = x.repeat([n_conds, 1, 1, 1])
    t_in = t.repeat([n_conds])
    clip_embed_in = torch.cat([*target_embeds]).repeat_interleave(n, 0)
    ts_mod = torch.jit.trace(model, (x_in, t_in, clip_embed_in))
    print(ts_mod.graph)

    clip_model = clip.load(clip_model_name, jit=True, device=device)[0]
    print(clip_model.graph)


# You can't run both of these because repro will `.half()` the model
# repro(model)
trace(model, x, t[0])
