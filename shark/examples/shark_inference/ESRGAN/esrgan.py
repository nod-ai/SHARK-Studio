from ast import arg
import os.path as osp
import glob
import cv2
import numpy as np
import torch

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from shark.shark_inference import SharkInference
import torch_mlir
import tempfile
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )
        fea = self.lrelu(
            self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


############### Parsing args #####################
import argparse

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

p.add_argument("--device", type=str, default="cpu", help="the device to use")
p.add_argument(
    "--mlir_loc",
    type=str,
    default=None,
    help="location of the model's mlir file",
)
args = p.parse_args()
###################################################


def inference(input_m):
    return model(input_m)


def load_mlir(mlir_loc):
    import os

    if mlir_loc == None:
        return None
    print(f"Trying to load the model from {mlir_loc}.")
    with open(os.path.join(mlir_loc)) as f:
        mlir_module = f.read()
    return mlir_module


def compile_through_fx(model, inputs, mlir_loc=None):
    module = load_mlir(mlir_loc)
    if module == None:
        fx_g = make_fx(
            model,
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
        )(inputs)

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

        print("Torchscript graph generated successfully")
        module = torch_mlir.compile(
            ts_g,
            inputs,
            torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=False,
            verbose=False,
        )

    mlir_model = str(module)
    func_name = "forward"
    shark_module = SharkInference(
        mlir_model, func_name, device=args.device, mlir_dialect="linalg"
    )
    shark_module.compile()

    return shark_module


model_path = "models/RRDB_ESRGAN_x4.pth"  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device("cpu")

test_img_folder = "InputImages/*"

model = RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print("Model path {:s}. \nTesting...".format(model_path))

if __name__ == "__main__":
    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(
            np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        ).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            shark_module = compile_through_fx(inference, img_LR)
            shark_output = shark_module.forward((img_LR,))
            shark_output = torch.from_numpy(shark_output)
            shark_output = (
                shark_output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            )
            esrgan_output = (
                model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            )
        # SHARK OUTPUT
        shark_output = np.transpose(shark_output[[2, 1, 0], :, :], (1, 2, 0))
        shark_output = (shark_output * 255.0).round()
        cv2.imwrite(
            "OutputImages/{:s}_rlt_shark_output.png".format(base), shark_output
        )
        print("Generated SHARK's output")
        # ESRGAN OUTPUT
        esrgan_output = np.transpose(esrgan_output[[2, 1, 0], :, :], (1, 2, 0))
        esrgan_output = (esrgan_output * 255.0).round()
        cv2.imwrite(
            "OutputImages/{:s}_rlt_esrgan_output.png".format(base),
            esrgan_output,
        )
        print("Generated ESRGAN's output")
