# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

import os

spec_root = os.path.abspath(SPECPATH)
shark_root = os.path.join(spec_root, "../..")
apps_root = os.path.join(spec_root, "../"
print(spec_root)
print(shark_root)
print(apps_root)

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
datas += copy_metadata('diffusers')
datas += copy_metadata('transformers')
datas += copy_metadata('omegaconf')
datas += copy_metadata('safetensors')
datas += collect_data_files('iree')
datas += collect_data_files('google-cloud-storage')
datas += collect_data_files('shark')
datas += collect_data_files('apps')
datas += [
         ( 'resources/prompts.json', 'resources' ),
         ( 'resources/model_db.json', 'resources' ),
         ( 'resources/opt_flags.json', 'resources' ),
         ( 'resources/base_model.json', 'resources' ),
         ]

binaries = []

block_cipher = None

a = Analysis(
    ['scripts/txt2img.py'],
    pathex=[spec_root, shark_root, apps_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=['apps', 'shark', 'shark.*', 'shark.shark_inference', 'shark_inference', 'iree.tools.core' ],
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
    name='shark_sd_cli',
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
