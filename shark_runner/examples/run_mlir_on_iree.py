import argparse
import pathlib

import numpy as np

from torch_mlir.ir import Context, Module
from iree_utils import get_iree_compiled_module

MLIR_TYPE_TO_NP_TYPE = {'f32': np.float32,
                        'i64': np.int64}


def load_mlir(filename: pathlib.Path) -> Module:
    with open(filename) as f:
        mlir_module_str = f.read()
    with Context() as _:
        mlir_module = Module.parse(mlir_module_str)
    return mlir_module


def get_random_array_with_type(mlir_tensor_type: str):
    start = 'tensor<'
    end = '>'
    assert mlir_tensor_type.startswith(start) and \
        mlir_tensor_type.endswith(end), \
        'Expected type to be a `tensor<...>` type'

    shape_and_dtype = mlir_tensor_type[len(start):-len(end)].split('x')
    mlir_dtype = shape_and_dtype[-1]
    assert mlir_dtype in MLIR_TYPE_TO_NP_TYPE, \
        f'Numpy equivalent of type {mlir_dtype} not known'
    dtype = MLIR_TYPE_TO_NP_TYPE[mlir_dtype]

    shape = shape_and_dtype[:-1]
    shape = map(lambda x: x.replace('?', '1'), shape)
    shape = list(map(int, shape))

    return (np.random.rand(*shape) * 100).astype(dtype)


def get_random_inputs_for(mlir_module):
    result = []
    for func in mlir_module.body.operations:
        if func.name.value != 'forward':
            continue

        for arg in func.arguments:
            type_str = str(arg.type)
            result.append(get_random_array_with_type(type_str))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compile and run Linalg-on-tensors graph on IREE')
    parser.add_argument('filename', type=pathlib.Path,
                        help='MLIR file of graph')
    parser.add_argument('--inputs', type=pathlib.Path, default=None,
                        help='.npz file with arrays to use as inputs')
    args = parser.parse_args()

    print('Loading mlir...')
    mlir_module = load_mlir(args.filename)
    print('Compiling mlir with IREE...')
    iree_module, _ = get_iree_compiled_module(mlir_module, 'cpu')

    if args.inputs:
        print('Loading inputs...')
        loaded_files = np.load(args.inputs)
        np_inps = [loaded_files[name] for name in loaded_files.files]
    else:
        print('Generating random inputs...')
        np_inps = get_random_inputs_for(mlir_module)

    print('Running IREE module...')
    results = iree_module(*np_inps)
    print('Results:')
    print(results)
