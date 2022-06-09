import numpy as np
import urllib.request

from PIL import Image


# Returns a sample image in the COCO 2017 dataset in uint8.
def generate_input(workdir, input_details):
    # We use an image of a bear since this is an easy example.
    img_path = "https://storage.googleapis.com/iree-model-artifacts/coco_2017_000000000285.jpg"
    local_path = "/".join([workdir, "coco_2017_000000000285.jpg"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = np.array(Image.open(local_path).resize((shape[1], shape[2])))
    return im.reshape(shape)
