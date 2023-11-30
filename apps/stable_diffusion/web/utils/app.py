import os
import sys
import webview
import webview.util
import socket

from contextlib import closing
from multiprocessing import Process

from apps.stable_diffusion.src import args


def webview2_installed():
    if sys.platform != "win32":
        return False

    # On windows we want to ensure we have MS webview2 available so we don't fall back
    # to MSHTML (aka ye olde Internet Explorer) which is deprecated by pywebview, and
    # apparently causes SHARK not to load in properly.

    # Checking these registry entries is how Microsoft says to detect a webview2 installation:
    # https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution
    import winreg

    path = r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"

    # only way can find if a registry entry even exists is to try and open it
    try:
        # check for an all user install
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            path,
            0,
            winreg.KEY_QUERY_VALUE | winreg.KEY_WOW64_64KEY,
        ) as registry_key:
            value, type = winreg.QueryValueEx(registry_key, "pv")

    # if it didn't exist, we want to continue on...
    except WindowsError:
        try:
            # ...to check for a current user install
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                path,
                0,
                winreg.KEY_QUERY_VALUE | winreg.KEY_WOW64_64KEY,
            ) as registry_key:
                value, type = winreg.QueryValueEx(registry_key, "pv")
        except WindowsError:
            value = None
    finally:
        return (value is not None) and value != "" and value != "0.0.0.0"


def window(address):
    from tkinter import Tk

    window = Tk()

    # get screen width and height of display and make it more reasonably
    # sized as we aren't making it full-screen or maximized
    width = int(window.winfo_screenwidth() * 0.81)
    height = int(window.winfo_screenheight() * 0.91)
    webview.create_window(
        "SHARK AI Studio",
        url=address,
        width=width,
        height=height,
        text_select=True,
    )
    webview.start(private_mode=False, storage_path=os.getcwd())


def usable_port():
    # Make sure we can actually use the port given in args.server_port. If
    # not ask the OS for a port and return that as our port to use.

    port = args.server_port

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            with closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as sock:
                sock.bind(("0.0.0.0", 0))
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return sock.getsockname()[1]

    return port


def launch(port):
    # setup to launch as an app if app mode has been requested and we're able
    # to do it, answering whether we succeeded.
    if args.ui == "app" and (sys.platform != "win32" or webview2_installed()):
        try:
            t = Process(target=window, args=[f"http://localhost:{port}"])
            t.start()
            return True
        except webview.util.WebViewException:
            return False
    else:
        return False
