import os
import sys
from apps.stable_diffusion.src import get_available_devices


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)


nodlogo_loc = resource_path("logos/nod-logo.png")
available_devices = get_available_devices()
