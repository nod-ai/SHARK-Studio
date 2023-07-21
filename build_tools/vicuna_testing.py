import os
from sys import executable
import subprocess
from apps.language_models.scripts import vicuna


def test_loop():
    precisions = ["fp16", "int8", "int4"]
    devices = ["cpu"]
    for precision in precisions:
        for device in devices:
            model = vicuna.UnshardedVicuna(device=device, precision=precision)
            model.compile()
            del model
