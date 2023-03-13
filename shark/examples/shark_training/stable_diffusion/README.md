# Stable Diffusion Fine Tuning

## Installation (Linux)

### Activate shark.venv Virtual Environment

```shell
source shark.venv/bin/activate

# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
```

## Install dependencies

### Run the following installation commands:
```
pip install -U git+https://github.com/huggingface/diffusers.git
pip install accelerate transformers ftfy
```

### Build torch-mlir with the following branch:

Please cherry-pick this branch of torch-mlir: https://github.com/vivekkhandelwal1/torch-mlir/tree/sd-ops
and build it locally. You can find the instructions for using locally build Torch-MLIR,
here: https://github.com/nod-ai/SHARK#how-to-use-your-locally-built-iree--torch-mlir-with-shark

## Run the Stable diffusion fine tuning

To run the model with the default set of images and params, run:
```shell
python stable_diffusion_fine_tuning.py
```
By default the training is run through the PyTorch path. If you want to train the model using the Torchdynamo path of Torch-MLIR, you need to specify `--use_torchdynamo=True`.

The default number of training steps are `2000`, which would take many hours to complete based on your system config. You can pass the smaller value with the arg `--training_steps`. You can specify the number of images to be sampled for the result with the `--num_inference_samples` arg. For the number of inference steps you can use `--inference_steps` flag.

For example, you can run the training for a limited set of steps via the dynamo path by using the following command:
```
python stable_diffusion_fine_tuning.py --training_steps=1 --inference_steps=1 --num_inference_samples=1 --train_batch_size=1 --use_torchdynamo=True
```

You can also specify the device to be used via the flag `--device`. The default value is `cpu`, for GPU execution you can specify `--device="cuda"`.
