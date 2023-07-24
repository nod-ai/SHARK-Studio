from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_submodules

import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

# python path for pyinstaller
pathex = [".", "./apps/language_models/langchain", "./apps/language_models/src/pipelines/minigpt4_utils"]

# datafiles for pyinstaller
datas = []
datas += collect_data_files("torch")
datas += copy_metadata("torch")
datas += copy_metadata("tqdm")
datas += copy_metadata("regex")
datas += copy_metadata("requests")
datas += copy_metadata("packaging")
datas += copy_metadata("filelock")
datas += copy_metadata("numpy")
datas += copy_metadata("importlib_metadata")
datas += copy_metadata("torch-mlir")
datas += copy_metadata("omegaconf")
datas += copy_metadata("safetensors")
datas += copy_metadata("Pillow")
datas += copy_metadata("sentencepiece")
datas += copy_metadata("pyyaml")
datas += collect_data_files("tokenizers")
datas += collect_data_files("tiktoken")
datas += collect_data_files("accelerate")
datas += collect_data_files("diffusers")
datas += collect_data_files("transformers")
datas += collect_data_files("pytorch_lightning")
datas += collect_data_files("opencv_python")
datas += collect_data_files("skimage")
datas += collect_data_files("gradio")
datas += collect_data_files("gradio_client")
datas += collect_data_files("iree")
datas += collect_data_files("google_cloud_storage")
datas += collect_data_files("shark")
datas += collect_data_files("timm", include_py_files=True)
datas += collect_data_files("tkinter")
datas += collect_data_files("webview")
datas += collect_data_files("sentencepiece")
datas += collect_data_files("jsonschema")
datas += collect_data_files("jsonschema_specifications")
datas += collect_data_files("cpuinfo")
datas += [
    ("src/utils/resources/prompts.json", "resources"),
    ("src/utils/resources/model_db.json", "resources"),
    ("src/utils/resources/opt_flags.json", "resources"),
    ("src/utils/resources/base_model.json", "resources"),
    ("web/ui/css/*", "ui/css"),
    ("web/ui/logos/*", "logos"),
    ("../language_models/src/pipelines/minigpt4_utils/configs/*", "minigpt4_utils/configs"),
    ("../language_models/src/pipelines/minigpt4_utils/prompts/*", "minigpt4_utils/prompts"),
]


# hidden imports for pyinstaller
hiddenimports = ["shark", "shark.shark_inference", "apps"]
hiddenimports += [x for x in collect_submodules("skimage") if "tests" not in x]
hiddenimports += [
    x for x in collect_submodules("transformers") if "tests" not in x
]
hiddenimports += [x for x in collect_submodules("iree") if "tests" not in x]
