from shark.shark_inference import SharkInference

import torch
import numpy as np
import sys
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions

torch.manual_seed(0)

vision_models = [
    "alexnet",
    "resnet101",
    "resnet18",
    "resnet50",
    "squeezenet1_0",
    "wide_resnet50_2",
    "mobilenet_v3_small",
]


def get_torch_model(modelname):
    if modelname in vision_models:
        return get_vision_model(modelname)
    else:
        return get_hf_model(modelname)


##################### Hugging Face LM Models ###################################


class HuggingFaceLanguage(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        import transformers as trf

        transformers_path = trf.__path__[0]
        hf_model_path = f"{transformers_path}/models/{hf_model_name}"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def get_hf_model(name):
    from transformers import (
        BertTokenizer,
        TFBertModel,
    )

    model = HuggingFaceLanguage(name)
    # TODO: Currently the test input is set to (1,128)
    test_input = torch.randint(2, (1, 128))
    actual_out = model(test_input)

    # fx_g = make_fx(
    #     model(test_input),
    #     decomposition_table=get_decompositions(
    #         [
    #             torch.ops.aten.embedding_dense_backward,
    #             torch.ops.aten.native_layer_norm_backward,
    #             torch.ops.aten.slice_backward,
    #             torch.ops.aten.select_backward,
    #             torch.ops.aten.norm.ScalarOpt_dim,
    #             torch.ops.aten.native_group_norm,
    #             torch.ops.aten.upsample_bilinear2d.vec,
    #             torch.ops.aten.split.Tensor,
    #             torch.ops.aten.split_with_sizes,
    #         ]
    #     ),
    # )(test_input)
    #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/fx/experimental/proxy_tensor.py", line 225, in wrapped
    #     t = dispatch_trace(wrap_key(f, args), concrete_args=tuple(phs),
    #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/fx/experimental/proxy_tensor.py", line 167, in dispatch_trace
    #     graph = tracer.trace(root, concrete_args)
    #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 559, in trace
    #     fn, args = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)
    #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 388, in create_args_for_root
    #     co = fn_for_analysis.__code__
    # AttributeError: 'Tensor' object has no attribute '__code__'. Did you mean: '__mod__'?
    return model, test_input, actual_out

    # fx_g = make_fx(
    #     model,
    #     decomposition_table=get_decompositions(
    #         [
    #             torch.ops.aten.embedding_dense_backward,
    #             torch.ops.aten.native_layer_norm_backward,
    #             torch.ops.aten.slice_backward,
    #             torch.ops.aten.select_backward,
    #             torch.ops.aten.norm.ScalarOpt_dim,
    #             torch.ops.aten.native_group_norm,
    #             torch.ops.aten.upsample_bilinear2d.vec,
    #             torch.ops.aten.split.Tensor,
    #             torch.ops.aten.split_with_sizes,
    #         ]
    #     ),
    # )
    # return fx_g, test_input, actual_out

    # # Traceback (most recent call last):
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/generate_sharktank.py", line 214, in <module>
    # #     save_torch_model(args.torch_model_csv)
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/generate_sharktank.py", line 74, in save_torch_model
    # #     mlir_importer.import_debug(
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark/shark_importer.py", line 163, in import_debug
    # #     imported_mlir = self.import_mlir(
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark/shark_importer.py", line 109, in import_mlir
    # #     return self._torch_mlir(is_dynamic, tracing_required), func_name
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark/shark_importer.py", line 74, in _torch_mlir
    # #     return get_torch_mlir_module(
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_utils.py", line 123, in get_torch_mlir_module
    # #     module = torch_mlir.compile(
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch_mlir/__init__.py", line 120, in compile
    # #     scripted = torch.jit.trace(model, tuple(example_args))
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/jit/_trace.py", line 795, in trace
    # #     traced = torch._C._create_function_from_trace(
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/fx/experimental/proxy_tensor.py", line 225, in wrapped
    # #     t = dispatch_trace(wrap_key(f, args), concrete_args=tuple(phs),
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/fx/experimental/proxy_tensor.py", line 167, in dispatch_trace
    # #     graph = tracer.trace(root, concrete_args)
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 559, in trace
    # #     fn, args = self.create_args_for_root(fn, isinstance(root, torch.nn.Module), concrete_args)
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/fx/_symbolic_trace.py", line 388, in create_args_for_root
    # #     co = fn_for_analysis.__code__
    # #   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1208, in __getattr__
    # #     raise AttributeError("'{}' object has no attribute '{}'".format(
    # # AttributeError: 'HuggingFaceLanguage' object has no attribute '__code__'. Did you mean: '__call__'?


################################################################################

##################### Torch Vision Models    ###################################


class VisionModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train(False)

    def forward(self, input):
        return self.model.forward(input)


def get_vision_model(torch_model):
    import torchvision.models as models

    vision_models_dict = {
        "alexnet": models.alexnet(pretrained=True),
        "resnet18": models.resnet18(pretrained=True),
        "resnet50": models.resnet50(pretrained=True),
        "resnet101": models.resnet101(pretrained=True),
        "squeezenet1_0": models.squeezenet1_0(pretrained=True),
        "wide_resnet50_2": models.wide_resnet50_2(pretrained=True),
        "mobilenet_v3_small": models.mobilenet_v3_small(pretrained=True),
    }
    if isinstance(torch_model, str):
        torch_model = vision_models_dict[torch_model]
    model = VisionModule(torch_model)
    test_input = torch.randn(1, 3, 224, 224)
    actual_out = model(test_input)
    return model, test_input, actual_out


################################################################################

# Utility function for comparing two tensors (torch).
def compare_tensors(torch_tensor, numpy_tensor):
    # setting the absolute and relative tolerance
    rtol = 1e-02
    atol = 1e-03
    # torch_to_numpy = torch_tensor.detach().numpy()
    return np.allclose(torch_tensor, numpy_tensor, rtol, atol)
