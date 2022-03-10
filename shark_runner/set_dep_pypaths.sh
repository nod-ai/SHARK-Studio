echo "Don't forget to activate your venv before calling this script";

export SCRIPTPATH=$(dirname "$(realpath BASH_SOURCE)");


if [ -z ${TORCH_MLIR_BUILD_DIR+x} ]; then echo "TORCH_MLIR_BUILD_DIR is unset"; exit 1; else echo "TORCH_MLIR_BUILD_DIR is set to '$TORCH_MLIR_BUILD_DIR'"; fi
if [ -z ${IREE_BUILD_DIR+x} ]; then echo "IREE_BUILD_DIR is unset"; exit 1;  else echo "IREE_BUILD_DIR is set to '$IREE_BUILD_DIR'"; fi
if [ -z ${SHARK_SAMPLES_DIR+x} ]; then echo "SHARK_SAMPLES_DIR is unset, using '$SCRIPTPATH'"; export SHARK_SAMPLES_DIR=$SCRIPTPATH; else echo "SHARK_SAMPLES_DIR is set to '$SHARK_SAMPLES_DIR'"; fi


export PYTHONPATH=${PYTHONPATH}:${IREE_BUILD_DIR}/compiler-api/python_package:${IREE_BUILD_DIR}/bindings/python
export PYTHONPATH=${PYTHONPATH}:${TORCH_MLIR_BUILD_DIR}/tools/torch-mlir/python_packages/torch_mlir:${TORCH_MLIR_BUILD_DIR}/../examples
export PYTHONPATH=${PYTHONPATH}:${SHARK_SAMPLES_DIR}


