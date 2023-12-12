from enum import IntEnum
import math
import sys
import os


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


nodlogo_loc = resource_path("logos/nod-logo.png")
nodicon_loc = resource_path("logos/nod-icon.png")


class HSLHue(IntEnum):
    RED = 0
    YELLOW = 60
    GREEN = 120
    CYAN = 180
    BLUE = 240
    MAGENTA = 300


def hsl_color(alpha: float, start, end):
    b = (end - start) * (alpha if alpha > 0 else 0)
    result = b + start

    # Return a CSS HSL string
    return f"hsl({math.floor(result)}, 80%, 35%)"
