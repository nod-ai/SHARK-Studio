from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_submodules

import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

# python path for pyinstaller
pathex = [
    ".",
]

# datafiles for pyinstaller
datas = []
datas += copy_metadata("torch")
datas += copy_metadata("tokenizers")
datas += copy_metadata("tqdm")
datas += copy_metadata("regex")
datas += copy_metadata("requests")
datas += copy_metadata("packaging")
datas += copy_metadata("filelock")
datas += copy_metadata("numpy")
datas += copy_metadata("importlib_metadata")
datas += copy_metadata("omegaconf")
datas += copy_metadata("safetensors")
datas += copy_metadata("Pillow")
datas += copy_metadata("sentencepiece")
datas += copy_metadata("pyyaml")
datas += copy_metadata("huggingface-hub")
datas += copy_metadata("gradio")
datas += copy_metadata("scipy")
datas += collect_data_files("torch")
datas += collect_data_files("tokenizers")
datas += collect_data_files("accelerate")
datas += collect_data_files("diffusers")
datas += collect_data_files("transformers")
datas += collect_data_files("gradio")
datas += collect_data_files("gradio_client")
datas += collect_data_files("iree", include_py_files=True)
datas += collect_data_files("shark", include_py_files=True)
datas += collect_data_files("tqdm")
datas += collect_data_files("tkinter")
datas += collect_data_files("sentencepiece")
datas += collect_data_files("jsonschema")
datas += collect_data_files("jsonschema_specifications")
datas += collect_data_files("cpuinfo")
datas += collect_data_files("scipy", include_py_files=True)
datas += [
    ("web/ui/css/*", "ui/css"),
    ("web/ui/js/*", "ui/js"),
    ("web/ui/logos/*", "logos"),
]


# hidden imports for pyinstaller
hiddenimports = ["shark", "apps"]
hiddenimports += [x for x in collect_submodules("gradio") if "tests" not in x]
hiddenimports += [
    x for x in collect_submodules("diffusers") if "tests" not in x
]
blacklist = ["tests", "convert"]
hiddenimports += [
    x
    for x in collect_submodules("transformers")
    if not any(kw in x for kw in blacklist)
]
hiddenimports += [x for x in collect_submodules("iree") if "tests" not in x]
hiddenimports += ["iree._runtime"]
hiddenimports += collect_submodules('scipy')