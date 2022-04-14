#!/bin/bash
source shark_venv/bin/activate

pip install -r requirements.txt

pip install --find-links https://github.com/llvm/torch-mlir/releases torch-mlir
pip install --find-links https://github.com/NodLabs/SHARK/releases iree-compiler iree-runtime
pip install git+https://github.com/pytorch/functorch.git
pip install .
