import argparse
import pathlib

import numpy as np

import transformers
from transformers import AutoModelForMaskedLM, BertConfig

import torch
import torch.utils._pytree as pytree
from torch import fx

from functorch.compile import  memory_efficient_fusion, get_decompositions, \
    default_partition

from torch_mlir_utils import get_torch_mlir_module


def change_fx_graph_return_to_tuple(fx_g: fx.GraphModule) -> fx.GraphModule:
    for node in fx_g.graph.nodes:
        if node.op == 'output':
            # output nodes always have one argument
            node_arg = node.args[0]
            if isinstance(node_arg, list):
                node.args = (tuple(node_arg),)
    fx_g.graph.lint()
    fx_g.recompile()
    return fx_g


def save_mlir_graph(ts_module, inps, filename: pathlib.Path):
    def hack_handle_rngs(mlir_module_str):
        """
        Replace Torch-MLIR RNGs with 0.

        Torch-MLIR RNGs are currently not supported in IREE.
        Here we simply remove them by replacing them with a
        constant of value 0. This is only done because currently
        we only care about getting graphs to run, not to get them
        to output the right values.
        """
        return mlir_module_str.replace(
            'torch_c.get_next_seed : () -> i64',
            'arith.constant 0 : i64')

    torch_mlir_module = get_torch_mlir_module(ts_module, inps,
                                              dynamic=False,
                                              tracing_required=False,
                                              from_aot=True)
    torch_mlir_module_str = hack_handle_rngs(str(torch_mlir_module))
    with open(filename, 'w') as output_file:
        output_file.write(torch_mlir_module_str)


def simple_ts_compiler(fx_g: fx.GraphModule, inps):
    fx_g = change_fx_graph_return_to_tuple(fx_g)
    f = torch.jit.script(fx_g)
    f = torch.jit.freeze(f.eval())
    torch.jit.save(f, '/tmp/graph_module.pt')
    f = torch.jit.load('/tmp/graph_module.pt')
    return f


def make_ts_compiler(out_mlir_filename: pathlib.Path,
                     out_np_inputs_filename: pathlib.Path):
    def compiler(fx_g: fx.GraphModule, inps):
        f = simple_ts_compiler(fx_g, inps)
        save_mlir_graph(f, inps, out_mlir_filename)

        np_inps = list(map(lambda x: x.detach().numpy(), inps))
        np.savez(out_np_inputs_filename, *np_inps)

        return f
    return compiler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Bert-training graphs and input arrays')
    parser.add_argument('--forward-mlir', type=pathlib.Path, required=True,
                        help='File to save forward pass in')
    parser.add_argument('--forward-inputs', type=pathlib.Path, required=True,
                        help='.npz file to save input arrays in')
    # TODO: Add support to torch-MLIR for returning None
    # parser.add_argument('--backward-mlir', type=pathlib.Path, required=True,
    # help='File to save backward pass in')
    # parser.add_argument('--backward-inputs', type=pathlib.Path, required=True,
    # help='.npz file to save input arrays in')
    args = parser.parse_args()

    pytree._register_pytree_node(
        transformers.modeling_outputs.MaskedLMOutput,
        lambda x: ([x.loss, x.logits], None),
        lambda values, _: transformers.modeling_outputs.MaskedLMOutput(
            loss=values[1], logits=values[1]
        ),
    )

    torch.manual_seed(0)
    config = BertConfig()
    model_type = AutoModelForMaskedLM
    device = 'cpu'
    dtype = torch.float

    model = model_type.from_config(config).to(device, dtype=dtype)
    input_size = (4, 512)
    input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
    decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)
    train_inputs = {'input_ids': input_ids, 'labels': decoder_ids}

    aot_model = memory_efficient_fusion(
        model,
        fw_compiler=make_ts_compiler(args.forward_mlir, args.forward_inputs),
        # TODO: Add support to torch-MLIR for returning None
        bw_compiler=simple_ts_compiler,
        partition_fn=default_partition,
        decompositions=get_decompositions(
            [torch.ops.aten.embedding_dense_backward,
             torch.ops.aten.native_layer_norm_backward])
    )

    aot_model(**train_inputs).loss.sum().backward()
