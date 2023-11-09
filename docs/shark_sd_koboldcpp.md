# Overview

In [1.47.2](https://github.com/LostRuins/koboldcpp/releases/tag/v1.47.2) [Koboldcpp](https://github.com/LostRuins/koboldcpp) added AUTOMATIC1111 integration for image generation. Since SHARK implements a small subset of the A1111 REST api, you can also use SHARK for this. This document gives a starting point for how to get this working.

## In Action

![preview](https://user-images.githubusercontent.com/121311569/280557602-bb97bad0-fdf5-4922-a2cc-4f327f2760db.jpg)

## Memory considerations

Since both Koboldcpp and SHARK will use VRAM on your graphic card(s) running both at the same time using the same card will impose extra limitations on the model size you can fully offload to the video card in Koboldcpp. For me, on a RX 7900 XTX on Windows with 24 GiB of VRAM, the limit was about a 13 Billion parameter model with Q5_K_M quantisation.

## Performance Considerations

When using SHARK for image generation, especially with Koboldcpp, you need to be aware that it is currently designed to pay a large upfront cost in time compiling and tuning the model you select, to get an optimal individual image generation time. You need to be the judge as to whether this trade-off is going to be worth it for your OS and hardware combination.

It means that the first time you run a particular Stable Diffusion model for a particular combination of image size, LoRA, and VAE, SHARK will spend *many minutes* - even on a beefy machaine with very fast graphics card with lots of memory - building that model combination just so it can save it to disk. It may even have to go away and download the model if it doesn't already have it locally. Once it has done its build of a model combination for your hardware once, it shouldn't need to do it again until you upgrade to a newer SHARK version, install different drivers or change your graphics hardware. It will just upload the files it generated the first time to your graphics card and proceed from there.

This does mean however, that on a brand new fresh install of SHARK that has not generated any images on a model you haven't selected before, the first image Koboldcpp requests may look like it is *never* going finish and that the whole process has broken. Be forewarned, make yourself a cup of coffee, and expect a lot of messages about compilation and tuning from SHARK in the terminal you ran it from.

## Setup SHARK and prerequisites:

 * Make sure you have suitable drivers for your graphics card installed. See the prerequisties section of the [README](https://github.com/nod-ai/SHARK#readme).
 * Download the latest SHARK studio .exe from [here](https://github.com/nod-ai/SHARK/releases) or follow the instructions in the [README](https://github.com/nod-ai/SHARK#readme) for an advanced, Linux or Mac install.
 * Run SHARK from terminal/PowerShell with the `--api` flag. Since koboldcpp also expects both CORS support and the image generator to be running on port `7860` rather than SHARK default of `8080`, also include both the `--api_cors_origin` flag with a suitable origin (use `="*"` to enable all origins) and `--server_port=7860` on the command line. (See the if you want to run SHARK on a different port)

```powershell
## Run the .exe in API mode, with CORS support, on the A1111 endpoint port:
.\node_ai_shark_studio_<date>_<ver>.exe --api --api_cors_origin="*"  --server_port=7860

## Run trom the base directory of a source clone of SHARK on Windows:
.\setup_venv.ps1
python .\apps\stable_diffusion\web\index.py --api --api_cors_origin="*"  --server_port=7860

## Run a the base directory of a source clone of SHARK on Linux:
./setup_venv.sh
source shark.venv/bin/activate
python ./apps/stable_diffusion/web/index.py --api --api_cors_origin="*"  --server_port=7860

## An example giving improved performance on AMD cards using vulkan, that runs on the same port as A1111
.\node_ai_shark_studio_20320901_2525.exe --api --api_cors_origin="*" --device_allocator="caching" --server_port=7860

## Since the api respects most applicable SHARK command line arguments for options not specified,
## or currently unimplemented by API, there might be some you want to set, as listed in `--help`
.\node_ai_shark_studio_20320901_2525.exe --help

## For instance, the example above, but with a a custom VAE specified
.\node_ai_shark_studio_20320901_2525.exe --api --api_cors_origin="*" --device_allocator="caching" --server_port=7860 --custom_vae="clearvae_v23.safetensors"

## An example with multiple specific CORS origins
python apps/stable_diffusion/web/index.py --api --api_cors_origin="koboldcpp.example.com:7001" --api_cors_origin="koboldcpp.example.com:7002" --server_port=7860
```

SHARK should start in server mode, and you should see something like this:

![SHARK API startup](https://user-images.githubusercontent.com/121311569/280556294-c3f7fc1a-c8e2-467d-afe6-365638d6823a.png)

* Note: When running in api mode with `--api`, the .exe will not function as a webUI. Thus, the address or port shown in the terminal output will only be useful for API requests.


## Configure Koboldcpp for local image generation:

* Get the latest [Koboldcpp](https://github.com/LostRuins/koboldcpp/releases) if you don't already have it. If you have a recent AMD card that has ROCm HIP [support for Windows](https://rocmdocs.amd.com/en/latest/release/windows_support.html#windows-supported-gpus) or [support for Linux](https://rocmdocs.amd.com/en/latest/release/gpu_os_support.html#linux-supported-gpus), you'll likely prefer [YellowRosecx's ROCm fork](https://github.com/YellowRoseCx/koboldcpp-rocm).
* Start Koboldcpp in another terminal/Powershell and setup your model configuration. Refer to the [Koboldcpp README](https://github.com/YellowRoseCx/koboldcpp-rocm) for more details on how to do this if this is your first time using Koboldcpp.
* Once the main UI has loaded into your browser click the settings button, go to the advanced tab, and then choose *Local A1111* from the generate images dropdown:

  ![Settings button location](https://user-images.githubusercontent.com/121311569/280556246-10692d79-e89f-4fdf-87ba-82f3d78ed49d.png)

  ![Advanced Settings with 'Local A1111' location](https://user-images.githubusercontent.com/121311569/280556234-6ebc8ba7-1469-442a-93a7-5626a094ddf1.png)

  *if you get an error here, see the next section [below](#connecting-to-shark-on-a-different-address-or-port)*

* A list of Stable Diffusion models available to your SHARK instance should now be listed in the box below *generate images*. The default value will usually be set to `stabilityai/stable-diffusion-2-1-base`. Choose the model you want to use for image generation from the list (but see [performance considerations](#performance-considerations)).
* You should now be ready to generate images, either by clicking the 'Add Img' button above the text entry box:

  ![Add Image Button](https://user-images.githubusercontent.com/121311569/280556161-846c7883-4a83-4458-a56a-bd9f93ca354c.png)

  ...or by selecting the 'Autogenerate' option in the settings:

  ![Setting the autogenerate images option](https://user-images.githubusercontent.com/121311569/280556230-ae221a46-ba68-499b-a519-c8f290bbbeae.png)

  *I often find that even if I have selected autogenerate I have to do an 'add img' to get things started off*

* There is one final piece of image generation configuration within Koboldcpp you might want to do. This is also in the generate images section of advanced settings. Here there is, not very obviously, a 'style' button:

  ![Selecting the 'styles' button](https://user-images.githubusercontent.com/121311569/280556694-55cd1c55-a059-4b54-9293-63d66a32368e.png)

  This will bring up a dialog box where you can enter a short text that will sent as a prefix to the Prompt sent to SHARK:

  ![Entering extra image styles](https://user-images.githubusercontent.com/121311569/280556172-4aab9794-7a77-46d7-bdda-43df570ad19a.png)


## Connecting to SHARK on a different address or port

If you didn't set the port to `--server_port=7860` when starting SHARK, or you are running it on different machine on your network than you are running Koboldcpp, or to where you are running the koboldcpp's kdlite client frontend, then you very likely got the following error:

  ![Can't find the A1111 endpoint error](https://user-images.githubusercontent.com/121311569/280555857-601f53dc-35e9-4027-9180-baa61d2393ba.png)

As long as SHARK is running correctly, this means you need to set the url and port to the correct values in Koboldcpp. For instance. to set the port that Koboldcpp looks for an image generator to SHARK's default port of 8080:

* Select the cog icon the Generate Images section of Advanced settings:

     ![Selecting the endpoint cog](https://user-images.githubusercontent.com/121311569/280555866-4287ecc5-f29f-4c03-8f5a-abeaf31b0442.png)

* Then edit the port number at the end of the url in the 'A1111 Endpoint Selection' dialog box to read 8080:

     ![Changing the endpoint port](https://user-images.githubusercontent.com/121311569/280556170-f8848b7b-6fc9-4cf7-80eb-5c312f332fd9.png)

* Similarly, when running SHARK on a different machine you will need to change host part of the endpoint url to the hostname or ip address where SHARK is running, similarly:

    ![Changing the endpoint hostname](https://user-images.githubusercontent.com/121311569/280556167-c6541dea-0f85-417a-b661-fdf4dc40d05f.png)

## Examples

Here's how Koboldcpp shows an image being requested:

  ![An image being generated]((https://user-images.githubusercontent.com/121311569/280556210-bb1c9efd-79ac-478e-b726-b25b82ef2186.png)

The generated image in context in story mode:

 ![A generated image](https://user-images.githubusercontent.com/121311569/280556179-4e9f3752-f349-4cba-bc6a-f85f8dc79b10.jpg)

And the same image when clicked on:

 ![A selected image](https://user-images.githubusercontent.com/121311569/280556216-2ca4c0a4-3889-4ef5-8a09-30084fb34081.jpg)


## Where to find the images in SHARK

Even though Koboldcpp requests images at a size of 512x512, it resizes then to 256x256, converts them to `.jpeg`, and only shows them at 200x200 in the main text window. It does this so it can save them compactly embedded in your story as a `data://` uri.

However the images at the original size are saved by SHARK in its `output_dir` which is usually a folder named for the current date. inside `generated_imgs` folder in the SHARK installation directory.

You can browse these, either using the Output Gallery tab from within the SHARK web ui:

  ![SHARK web ui output gallery tab](https://user-images.githubusercontent.com/121311569/280556582-9303ca85-2594-4a8c-97a2-fbd72337980b.jpg)

...or by browsing to the `output_dir` in your operating system's file manager:

  ![SHARK output directory subfolder in Windows File Explorer](https://user-images.githubusercontent.com/121311569/280556297-66173030-2324-415c-a236-ef3fcd73e6ed.jpg)
