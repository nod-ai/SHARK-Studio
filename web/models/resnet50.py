from PIL import Image
import requests
import torch
from torchvision import transforms
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_model

################################## Preprocessing inputs and helper functions ########

DEBUG = False
compiled_module = {}


def preprocess_image(img):
    image = Image.fromarray(img)
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
    img_preprocessed = preprocess(image)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels():
    classes_text = requests.get(
        "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt",
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels


def top3_possibilities(res, log_write):
    global DEBUG

    if DEBUG:
        log_write.write("Retrieving top 3 possible outcomes.\n")
    labels = load_labels()
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0]
    top3 = dict(
        [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    )
    if DEBUG:
        log_write.write("Done.\n")
    return top3


##############################################################################


def resnet_inf(numpy_img, device):
    global DEBUG
    global compiled_module

    DEBUG = False
    log_write = open(r"logs/resnet50_log.txt", "w")
    if log_write:
        DEBUG = True

    if device not in compiled_module.keys():
        if DEBUG:
            log_write.write("Compiling the Resnet50 module.\n")
        mlir_model, func_name, inputs, golden_out = download_model(
            "resnet50", frontend="torch"
        )
        shark_module = SharkInference(
            mlir_model, func_name, device=device, mlir_dialect="linalg"
        )
        shark_module.compile()
        compiled_module[device] = shark_module
        if DEBUG:
            log_write.write("Compilation successful.\n")

    img = preprocess_image(numpy_img)
    result = compiled_module[device].forward((img.detach().numpy(),))
    output = top3_possibilities(torch.from_numpy(result), log_write)
    log_write.close()

    std_output = ""
    with open(r"logs/resnet50_log.txt", "r") as log_read:
        std_output = log_read.read()

    return output, std_output
