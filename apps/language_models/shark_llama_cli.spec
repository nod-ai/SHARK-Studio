# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata

import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

datas = []
datas += collect_data_files('torch')
datas += copy_metadata('torch')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('importlib_metadata')
datas += copy_metadata('torch-mlir')
datas += copy_metadata('omegaconf')
datas += copy_metadata('safetensors')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('sentencepiece')
datas += copy_metadata("pyyaml")
datas += collect_data_files("tokenizers")
datas += collect_data_files("tiktoken")
datas += collect_data_files("accelerate")
datas += collect_data_files('diffusers')
datas += collect_data_files('transformers')
datas += collect_data_files('opencv-python')
datas += collect_data_files('pytorch_lightning')
datas += collect_data_files('skimage')
datas += collect_data_files('gradio')
datas += collect_data_files('gradio_client')
datas += collect_data_files('iree')
datas += collect_data_files('google-cloud-storage')
datas += collect_data_files('py-cpuinfo')
datas += collect_data_files("shark", include_py_files=True)
datas += collect_data_files("timm", include_py_files=True)
datas += collect_data_files("tqdm")
datas += collect_data_files("tkinter")
datas += collect_data_files("webview")
datas += collect_data_files("sentencepiece")
datas += collect_data_files("jsonschema")
datas += collect_data_files("jsonschema_specifications")
datas += collect_data_files("cpuinfo")
datas += collect_data_files("langchain")

binaries = []

block_cipher = None

hiddenimports = ['shark', 'shark.shark_inference', 'apps']
hiddenimports += [x for x in collect_submodules("skimage") if "tests" not in x]
hiddenimports += [x for x in collect_submodules("iree") if "tests" not in x]

a = Analysis(
    ['scripts/vicuna.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='shark_llama_cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
