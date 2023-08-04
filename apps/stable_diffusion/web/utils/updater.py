# This file contains auto update triggers for all installable distributions
import os
import sys
import subprocess


# In Windows MSI ONLY
def install_updates():
    """Installs update from latest release on GitHub if currently running on Windows MSI and an update exists"""

    # Ensures that code is running in a pyinstaller bundle on windows
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return

    # Gets path to updater.exe
    application_path = os.path.dirname(sys.executable)
    application_path, _ = os.path.split(application_path)
    application_path = os.path.join(application_path, "updater.exe")

    # If updater exe does not exist, do nothing
    if not os.path.isfile(application_path):
        return

    # run updater.exe
    subprocess.run(application_path)
