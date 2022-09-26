from PIL import Image
import requests
import torch
from torchvision import transforms
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_torch_model

################################## Preprocessing inputs and helper functions ########


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


def top3_possibilities(res):
    labels = load_labels()
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0]
    top3 = dict(
        [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    )
    return top3


##############################################################################

compiled_module = {}


def resnet_inf(numpy_img, device):

    global compiled_module

    std_output = ""

    if device not in compiled_module.keys():
        std_output += "Compiling the Resnet50 module.\n"
        mlir_model, func_name, inputs, golden_out = download_torch_model(
            "resnet50"
        )

        shark_module = SharkInference(
            mlir_model, func_name, device=device, mlir_dialect="linalg"
        )
        shark_module.compile()
        std_output += "Compilation successful.\n"
        compiled_module[device] = shark_module

    img = preprocess_image(numpy_img)
    result = compiled_module[device].forward((img.detach().numpy(),))

    #  print("The top 3 results obtained via shark_runner is:")
    std_output += "Retrieving top 3 possible outcomes.\n"
    return top3_possibilities(torch.from_numpy(result)), std_output
