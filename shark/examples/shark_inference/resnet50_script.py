from PIL import Image
import requests
import torch
import torchvision.models as models
from torchvision import transforms
import sys
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_torch_model
import threading
from cuda import cudart
import copy
import numpy as np
import time

################################## Preprocessing inputs and model ############
def load_and_preprocess_image(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }
    img = Image.open(
        requests.get(url, headers=headers, stream=True).raw
    ).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels():
    classes_text = requests.get(
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt",
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels


def top3_possibilities(res):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3


class Resnet50Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


image_url = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
print("load image from " + image_url, file=sys.stderr)
img = load_and_preprocess_image(image_url)
labels = load_labels()

##############################################################################


## Can pass any img or input to the forward module.

mlir_model, func_name, inputs, golden_out = download_torch_model("resnet50")

def thread_function(device_idx):

    cudart.cudaSetDevice(device_idx)
    print('in thread {}'.format(device_idx))
    
    shark_module = SharkInference(mlir_model, func_name,  mlir_dialect="linalg", device_idx=device_idx)
    shark_module.compile()
    print('device {} compiled'.format(device_idx))
    in_ = (img.detach().numpy(),)

    result = shark_module.forward(in_)

    print(top3_possibilities(torch.from_numpy(result)))
    print('Device {} executed successfully'.format(device_idx))

x1 = threading.Thread(target=thread_function, args =(0,))
x2 = threading.Thread(target=thread_function, args = (1,))
x1.start()
x2.start()


