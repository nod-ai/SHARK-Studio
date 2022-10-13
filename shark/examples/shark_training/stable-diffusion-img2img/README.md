# Stable Diffusion Img2Img model

## Installation

<details>
  <summary>Installation (Linux)</summary>

### Activate shark.venv Virtual Environment

```shell
source shark.venv/bin/activate

# Some older pip installs may not be able to handle the recent PyTorch deps
python -m pip install --upgrade pip
```

### Install dependencies

# Run the setup.sh script

```shell
./setup.sh
```

### Run the Stable diffusion Img2Img model

To run the model with the default set of images and params, run:
```shell
python stable_diffusion_img2img.py
```
To run the model with your set of images, and parameters you need to specify the following params:
1.) Input images directory with the arg `--input_dir` containing 3-5 images.
2.) What to teach the model? Using the arg `--what_to_teach`, allowed values are `object` or `style`.
3.) Placeholder token using the arg `--placeholder_token`, that represents your new concept. It should be passed with the opening and closing angle brackets. For ex: token is `cat-toy`, it should be passed as `<cat-toy>`.
4.) Initializer token using the arg `--initializer_token`, which summarise what is your new concept.

For the result, you need to pass the text prompt with the arg: `--prompt`. The prompt string should contain a "*s" in it, which will be replaced by the placeholder token during the inference.

By default the result images will go into the `sd_result` dir. To specify your output dir use the arg: `--output_dir`.

The default value of max_training_steps is `3000`, which takes some hours to complete. You can pass the smaller value with the arg `--training_steps`. Specify the number of images to be sampled for the result with the `--num_inference_samples` arg.
