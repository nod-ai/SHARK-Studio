# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
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
datas += copy_metadata('torchvision')
datas += copy_metadata('torch-mlir')
datas += copy_metadata('diffusers')
datas += copy_metadata('transformers')
datas += collect_data_files('gradio')
datas += collect_data_files('iree')
datas += collect_data_files('google-cloud-storage')
datas += collect_data_files('shark')
datas += [
         ( 'models/stable_diffusion/resources/prompts.json', 'resources' ),
         ( 'models/stable_diffusion/resources/model_db.json', 'resources' ),
         ( 'models/stable_diffusion/resources/model_config.json', 'resources' ),
         ( 'models/stable_diffusion/logos/*', 'logos' )
         ]
datas += [('demo.css', '.')]
binaries = []

block_cipher = None


a = Analysis(
    ['index.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['shark', 'shark.*', 'shark.shark_inference', 'shark_inference', 'iree.tools.core', 'gradio'],
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
    name='shark_sd',
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
