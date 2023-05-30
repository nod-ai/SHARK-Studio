## Run MiniGPT4 via WebUI

- Step 1: Run `pip install iopath timm webdataset decord`.
- Step 2: Download `vicuna_weights` and `prerained_minigpt4_7b.pth` from [MiniGPT4Demo in SharkTank](https://console.cloud.google.com/storage/browser/shark_tank/MiniGPT4Demo;tab=objects?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=0&prefix=&forceOnObjectsSortingFiltering=false).
- Step 3: Modify `ckpt` in [minigpt4_eval.yaml](https://github.com/Abhishek-Varma/SHARK/blob/wip_minigpt4/apps/stable_diffusion/web/ui/minigpt4/configs/minigpt4_eval.yaml) to point to path of `prerained_minigpt4_7b.pth` downloaded above.
- Step 4: Modify `llama_model` in [minigpt4.yaml](https://github.com/Abhishek-Varma/SHARK/blob/wip_minigpt4/apps/stable_diffusion/web/ui/minigpt4/configs/minigpt4.yaml) to point to path of `vicuna_weights` downloaded above.
- Run `python apps/stable_diffusion/web/index.py` and chat with MiniGPT4!