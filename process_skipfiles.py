# This script will toggle the comment/uncommenting aspect for dealing
# with __file__ AttributeError arising in case of a few modules in
# `torch/_dynamo/skipfiles.py` (within shark.venv)

from distutils.sysconfig import get_python_lib
import fileinput
from pathlib import Path

# Temorary workaround for transformers/__init__.py.
path_to_tranformers_hook = Path(
    get_python_lib()
    + "/_pyinstaller_hooks_contrib/hooks/stdhooks/hook-transformers.py"
)
if path_to_tranformers_hook.is_file():
    pass
else:
    with open(path_to_tranformers_hook, "w") as f:
        f.write("module_collection_mode = 'pyz+py'")

path_to_skipfiles = Path(get_python_lib() + "/torch/_dynamo/skipfiles.py")

modules_to_comment = ["abc,", "os,", "posixpath,", "_collections_abc,"]
startMonitoring = 0
for line in fileinput.input(path_to_skipfiles, inplace=True):
    if "SKIP_DIRS = " in line:
        startMonitoring = 1
        print(line, end="")
    elif startMonitoring in [1, 2]:
        if "]" in line:
            startMonitoring += 1
            print(line, end="")
        else:
            flag = True
            for module in modules_to_comment:
                if module in line:
                    if not line.startswith("#"):
                        print(f"#{line}", end="")
                    else:
                        print(f"{line[1:]}", end="")
                    flag = False
                    break
            if flag:
                print(line, end="")
    else:
        print(line, end="")

# For getting around scikit-image's packaging, laze_loader has had a patch merged but yet to be released.
# Refer: https://github.com/scientific-python/lazy_loader
path_to_lazy_loader = Path(get_python_lib() + "/lazy_loader/__init__.py")

for line in fileinput.input(path_to_lazy_loader, inplace=True):
    if 'stubfile = filename if filename.endswith("i")' in line:
        print(
            '    stubfile = (filename if filename.endswith("i") else f"{os.path.splitext(filename)[0]}.pyi")',
            end="",
        )
    else:
        print(line, end="")

# For getting around timm's packaging.
# Refer: https://github.com/pyinstaller/pyinstaller/issues/5673#issuecomment-808731505
path_to_timm_activations = Path(
    get_python_lib() + "/timm/layers/activations_jit.py"
)
for line in fileinput.input(path_to_timm_activations, inplace=True):
    if "@torch.jit.script" in line:
        print("@torch.jit._script_if_tracing", end="\n")
    else:
        print(line, end="")