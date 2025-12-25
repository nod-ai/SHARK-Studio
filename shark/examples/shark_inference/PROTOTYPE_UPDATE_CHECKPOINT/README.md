## Running PROTOTYPE UPDATE CHECKPOINT WEIGHT

- Since this is a prototype, you need to clone and build [Abhishek-torch-mlir](https://github.com/Abhishek-Varma/torch-mlir.git).
- We'd be using branch `prototype_update_weight`, so you may use the command : `git clone https://github.com/Abhishek-Varma/torch-mlir.git --branch prototype_update_weight` to clone and checkout the branch at the same time.
- Follow the build procedure of [SHARK](https://github.com/Abhishek-Varma/SHARK/blob/main/README.md).
- Then run `pip uninstall torch-mlir` followed by `export PYTHONPATH=~/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir:~/torch-mlir/examples:$PYTHONPATH` (assuming you've clone the `torch-mlir` file in your home directory itself). These commands will ensure you're using the locally built `torch-mlir` for SHARK.
- Run : `python mynetwork.py` (You might need to update the path to the initial `linalgIR.mlir` file for now)
- NOTE: This README.md will be updated once SharkInference is indeed used.
