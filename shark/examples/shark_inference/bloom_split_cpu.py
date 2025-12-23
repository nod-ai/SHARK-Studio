from shark.shark_inference import SharkInference
from shark.shark_downloader import download_torch_model
from shark.sharding_utils.split_mlir_file import split_mlir_file
from pathlib import Path
import os
import torch
import numpy as np

download_torch_model("bloom")

home = str(Path.home())
WORKDIR = os.path.join(home, ".local/shark_tank/bloom_splits/")

if not os.path.exists(os.path.join(home, ".local/shark_tank/bloom_splits")):
    os.makedirs(os.path.join(home, ".local/shark_tank/bloom_splits"))

replace = False
n_splits = 10
# Split Bloom_mlir File

for i in range(0, n_splits):
    if replace or not os.path.exists(
        os.path.join(
            home, ".local/shark_tank/bloom_splits/bloom{}.mlir".format(i)
        )
    ):
        split_mlir_file(
            os.path.join(
                home, ".local/shark_tank/bloom_torch/bloom_torch.mlir"
            ),
            i * 1000,
            (i + 1) * 1000,
            os.path.join(
                home, ".local/shark_tank/bloom_splits/bloom{}.mlir".format(i)
            ),
            first=(i == 0),
            last=(i == n_splits - 1),
        )

input = torch.tensor(np.zeros([1, 1024]))
input = input.to(torch.int64)
input = [input]


def run_mlir_split(n, input):

    f_ = open(
        os.path.join(
            home, ".local/shark_tank/bloom_splits/bloom{}.mlir".format(n)
        )
    )
    mlir_model = f_.read()
    f_.close()

    shark_module = SharkInference(
        mlir_model, "forward", mlir_dialect="tm_tensor"
    )
    shark_module.compile()
    result = shark_module.forward(tuple(input))
    print(f"Bloom {n} Successful")
    return result


for x in range(0, 10):
    input = run_mlir_split(x, input)
result = input
print(result)
