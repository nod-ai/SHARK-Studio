import logging
import os
import pathlib
import sys

logger = logging.getLogger(__name__)

# App info
APP_NAME = "shark_sd"
APP_VERSION = 12082022.365

# Current module dir (when frozen this equals sys._MEIPASS)
# https://pyinstaller.org/en/stable/runtime-information.html#using-file
MODULE_DIR = pathlib.Path(__file__).resolve().parent

# Are we running in a PyInstaller bundle?
# https://pyinstaller.org/en/stable/runtime-information.html
FROZEN = getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")

# For development
DEV_DIR = MODULE_DIR 

# App directories
PROGRAMS_DIR = pathlib.Path(sys._MEIPASS if FROZEN else DEV_DIR)
DATA_DIR = pathlib.Path(sys._MEIPASS if FROZEN else DEV_DIR)
INSTALL_DIR = pathlib.Path(sys._MEIPASS if FROZEN else DEV_DIR)
UPDATE_CACHE_DIR = pathlib.Path(
    os.path.join(DATA_DIR, "update_cache")
)
METADATA_DIR = pathlib.Path(os.path.join(UPDATE_CACHE_DIR, "metadata"))
TARGET_DIR = pathlib.Path(os.path.join(UPDATE_CACHE_DIR, "targets"))

# Update-server urls
METADATA_BASE_URL = (
    "https://storage.googleapis.com/shark_tank/releases/metadata"
)
TARGET_BASE_URL = "https://storage.googleapis.com/shark_tank/releases/targets"

# Location of trusted root metadata file
TRUSTED_ROOT_SRC = MODULE_DIR / "root.json"
if not FROZEN:
    # for development, get the root metadata directly from local repo
    TRUSTED_ROOT_SRC = (
        MODULE_DIR.parent / "temp/repository/metadata/root.json"
    )
TRUSTED_ROOT_DST = TRUSTED_ROOT_SRC / "root.json"
