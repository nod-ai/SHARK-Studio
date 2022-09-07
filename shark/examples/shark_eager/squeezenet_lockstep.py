import torch
import numpy as np

model = torch.hub.load(
    "pytorch/vision:v0.10.0", "squeezenet1_0", pretrained=True
)
model.eval()

# from PIL import Image
# from torchvision import transforms
# import urllib
#
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)
#
#
# input_image = Image.open(filename)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
# print(input_batch.shape) # size = [1, 3, 224, 224]

# The above is code for generating sample inputs from an image. We can just use
# random values for accuracy testing though
input_batch = torch.randn(1, 3, 224, 224)


# Focus on CPU for now
if False and torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
golden_confidences = output[0]
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
golden_probabilities = torch.nn.functional.softmax(
    golden_confidences, dim=0
).numpy()

golden_confidences = golden_confidences.numpy()

from shark.torch_mlir_lockstep_tensor import TorchMLIRLockstepTensor

input_detached_clone = input_batch.clone()
eager_input_batch = TorchMLIRLockstepTensor(input_detached_clone)

print("getting torch-mlir result")

output = model(eager_input_batch)

static_output = output.elem
confidences = static_output[0]
probabilities = torch.nn.functional.softmax(
    torch.from_numpy(confidences), dim=0
).numpy()

print("The obtained result via shark is: ", confidences)
print("The golden result is:", golden_confidences)

np.testing.assert_allclose(
    golden_confidences, confidences, rtol=1e-02, atol=1e-03
)
np.testing.assert_allclose(
    golden_probabilities, probabilities, rtol=1e-02, atol=1e-03
)
