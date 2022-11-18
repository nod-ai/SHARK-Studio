import imp
from PIL import Image
import cv2
import numpy as np
import requests
import torch
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
import time
from ast import arg
import os
import os.path as osp
import glob
import torch
from torch._decomp import get_decompositions
import torch_mlir
import tempfile
import functools
import torch.nn as nn
import torch.nn.functional as F

############### Model ################
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


#####################################


def process_image(img):
    img = img * 1.0 / 255
    img = torch.from_numpy(
        np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
    ).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    return img_LR


def inference(input_m):
    return model(input_m)


def compile_through_fx(model, inputs, device, mlir_loc=None):
    mlir_module, func_name = import_with_fx(model, (inputs,))
    shark_module = SharkInference(
        mlir_module, func_name, device=device, mlir_dialect="linalg"
    )
    shark_module.compile()

    return shark_module


DEBUG = False
compiled_module = {}
model_path = "RRDB_ESRGAN_x4.pth"  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device("cpu")

model = RRDBNet(3, 3, 64, 23, gc=32)


def esrgan_inf(numpy_img, device):
    global model
    model = RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    global DEBUG
    global compiled_module

    DEBUG = False
    log_write = open(r"logs/esrgan_log.txt", "w")
    if log_write:
        DEBUG = True

    if DEBUG:
        log_write.write("Compiling the ESRGAN module.\n")
    numpy_img = np.flip(numpy_img, 2)
    img_LR = process_image(numpy_img)
    with torch.no_grad():
        shark_module = compile_through_fx(inference, img_LR, device)
        if DEBUG:
            log_write.write("Compilation successful.\n")
        shark_output = shark_module.forward((img_LR,))
        shark_output = torch.from_numpy(shark_output)
        shark_output = (
            shark_output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        )
    shark_output = np.transpose(shark_output[[2, 1, 0], :, :], (1, 2, 0))
    output = np.flip(shark_output, 2)
    shark_output = (shark_output * 255.0).round()
    cv2.imwrite(
        f"stored_results/esrgan/{str(int(time.time()))}_{device}.png",
        shark_output,
    )
    print("Upscaled image generated")
    if DEBUG:
        log_write.write("ESRGAN upscaler ran successfully\n")
    log_write.close()

    std_output = ""
    with open(r"logs/esrgan_log.txt", "r") as log_read:
        std_output = log_read.read()

    return output, std_output
