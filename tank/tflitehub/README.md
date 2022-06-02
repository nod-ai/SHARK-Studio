# Sample compile and execution of TFLite models

This directory contains test scripts to compile/run/compare various TFLite
models from TFHub. It aims for simplicity and hackability.

Follow the instructions at the repository root to install a functioning
python venv. Then you can just run individual python files.

Or, use something like the following to collect all artifacts and traces,
which can be fed to other tools:

```
export IREE_SAVE_TEMPS="/tmp/iree/models/{main}/{id}"
for i in *.py; do export IREE_SAVE_CALLS=/tmp/iree/traces/$i; python $i; done
```
