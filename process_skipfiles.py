# This script will toggle the comment/uncommenting aspect for dealing
# with __file__ AttributeError arising in case of a few modules in
# `torch/_dynamo/skipfiles.py` (within shark.venv)

from distutils.sysconfig import get_python_lib
import fileinput
from pathlib import Path
import os

# Temporary workaround for transformers/__init__.py.
path_to_transformers_hook = Path(
    get_python_lib() + "/_pyinstaller_hooks_contrib/hooks/stdhooks/hook-transformers.py"
)
if path_to_transformers_hook.is_file():
    pass
else:
    with open(path_to_transformers_hook, "w") as f:
        f.write("module_collection_mode = 'pyz+py'")

paths_to_skipfiles = [Path(get_python_lib() + "/torch/_dynamo/skipfiles.py"), Path(get_python_lib() + "/torch/_dynamo/trace_rules.py")]

for path in paths_to_skipfiles:
    if not os.path.isfile(path):
        continue
    for line in fileinput.input(path, inplace=True):
        if "[_module_dir(m) for m in BUILTIN_SKIPLIST]" in line and "x.__name__ for x in BUILTIN_SKIPLIST" not in line:
            print(f"{line.rstrip()} + [x.__name__ for x in BUILTIN_SKIPLIST]")
        elif "(_module_dir(m) for m in BUILTIN_SKIPLIST)" in line and "x.__name__ for x in BUILTIN_SKIPLIST" not in line:
            print(line, end="")
            print(f"SKIP_DIRS.extend(filter(None, (x.__name__ for x in BUILTIN_SKIPLIST)))")
        else:
            print(line, end="")
