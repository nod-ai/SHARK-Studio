# Lint as: python3
"""SHARK Importer"""

import sys
import tempfile
import os

# List of the supported frontends.
supported_frontends = {
    "tensorflow",
    "tf",
    "pytorch",
    "torch",
    "tf-lite",
    "tflite",
}


class SharkImporter:
    """
    SharkImporter converts frontend modules into a
    mlir_module. The supported frameworks are tensorflow,
    pytorch, and tf-lite.

    ...

    Attributes
    ----------
    module :
        torch, tensorflow or tf-lite module.
    inputs :
        inputs to the module, may be required for the shape
        information.
    frontend: str
        frontend to which the module belongs.
    raw_model_file: str
        temp tflite model path

    Methods
    -------
    import_mlir(is_dynamic, tracing_required, func_name):
        is_dynamic: input shapes to be totally dynamic (pytorch specific).
        tracing_required: whether tracing is required (pytorch specific.
        func_name: The function to be traced out or imported to mlir.

    import_debug(is_dynamic, tracing_required, func_name):
        returns the converted (mlir_module,func_name) with inputs and golden
        outputs.
        The inputs and outputs are converted into np array.
    """

    def __init__(
        self,
        module,
        inputs: tuple = (),
        frontend: str = "torch",
        raw_model_file: str = "",
    ):
        self.module = module
        self.inputs = None if len(inputs) == 0 else inputs
        self.frontend = frontend
        if not self.frontend in supported_frontends:
            print(
                f"The frontend is not in the supported_frontends: {supported_frontends}"
            )
            sys.exit(1)
        self.raw_model_file = raw_model_file

    # NOTE: The default function for torch is "forward" and tf-lite is "main".

    def _torch_mlir(self, is_dynamic, tracing_required):
        from shark.torch_mlir_utils import get_torch_mlir_module

        return get_torch_mlir_module(
            self.module, self.inputs, is_dynamic, tracing_required
        )

    def _tf_mlir(self, func_name):
        from iree.compiler import tf as tfc

        return tfc.compile_module(
            self.module, exported_names=[func_name], import_only=True
        )

    def _tflite_mlir(self, func_name):
        from iree.compiler import tflite as tflitec
        from shark.iree_utils._common import IREE_TARGET_MAP

        self.mlir_model = tflitec.compile_file(
            self.raw_model_file,  # in tflite, it is a path to .tflite file, not a tflite interpreter
            input_type="tosa",
            import_only=True,
        )
        return self.mlir_model

    # Adds the conversion of the frontend with the private function.
    def import_mlir(
        self,
        is_dynamic=False,
        tracing_required=False,
        func_name="forward",
    ):
        if self.frontend in ["torch", "pytorch"]:
            if self.inputs == None:
                print(
                    "Please pass in the inputs, the inputs are required to determine the shape of the mlir_module"
                )
                sys.exit(1)
            return self._torch_mlir(is_dynamic, tracing_required), func_name
        if self.frontend in ["tf", "tensorflow"]:
            return self._tf_mlir(func_name), func_name
        if self.frontend in ["tflite", "tf-lite"]:
            func_name = "main"
            return self._tflite_mlir(func_name), func_name

    # Converts the frontend specific tensors into np array.
    def convert_to_numpy(self, array_tuple: tuple):
        if self.frontend in ["torch", "pytorch"]:
            return [x.detach().cpu().numpy() for x in array_tuple]
        if self.frontend in ["tf", "tensorflow"]:
            return [x.numpy() for x in array_tuple]

    # Saves `function_name.npy`, `inputs.npz`, `golden_out.npz` and `model_name.mlir` in the directory `dir`.
    def save_data(
        self, dir, model_name, mlir_data, func_name, inputs, outputs
    ):
        import numpy as np

        inputs_name = "inputs.npz"
        outputs_name = "golden_out.npz"
        func_file_name = "function_name"
        model_name_mlir = model_name + "_" + self.frontend + ".mlir"
        inputs = [x.cpu().detach() for x in inputs]
        np.savez(os.path.join(dir, inputs_name), *inputs)
        np.savez(os.path.join(dir, outputs_name), *outputs)
        np.save(os.path.join(dir, func_file_name), np.array(func_name))

        mlir_str = mlir_data
        if self.frontend == "torch":
            mlir_str = mlir_data.operation.get_asm()
        elif self.frontend == "tf":
            try:
                mlir_str = mlir_data.decode("utf-8")
            except:
                mlir_str = mlir_data.decode("latin-1")
        elif self.frontend == "tflite":
            try:
                mlir_str = mlir_data.decode("utf-8")
            except:
                mlir_str = mlir_data.decode("latin-1")
        with open(os.path.join(dir, model_name_mlir), "w") as mlir_file:
            mlir_file.write(mlir_str)

        return

    def import_debug(
        self,
        is_dynamic=False,
        tracing_required=False,
        func_name="forward",
        dir=tempfile.gettempdir(),
        model_name="model",
    ):
        if self.inputs == None:
            print(
                f"There is no input provided: {self.inputs}, please provide inputs or simply run import_mlir."
            )
            sys.exit(1)

        imported_mlir = self.import_mlir(
            is_dynamic, tracing_required, func_name
        )
        # TODO: Make sure that any generic function name is accepted. Currently takes in the default function names.
        # TODO: Check for multiple outputs.
        if self.frontend in ["torch", "pytorch"]:
            import torch

            golden_out = self.module(*self.inputs)
            if torch.is_tensor(golden_out):
                golden_out = tuple(
                    golden_out.detach().cpu().numpy(),
                )
            else:
                golden_out = self.convert_to_numpy(golden_out)
            # Save the artifacts in the directory dir.
            self.save_data(
                dir,
                model_name,
                imported_mlir[0],
                imported_mlir[1],
                self.inputs,
                golden_out,
            )
            return (
                imported_mlir,
                self.convert_to_numpy(self.inputs),
                golden_out,
            )
        if self.frontend in ["tf", "tensorflow"]:
            import tensorflow as tf

            golden_out = self.module.forward(*self.inputs)
            if tf.is_tensor(golden_out):
                golden_out = tuple(
                    golden_out.numpy(),
                )
            elif golden_out is tuple:
                golden_out = self.convert_to_numpy(golden_out)
            elif hasattr(golden_out, "logits"):
                # from transformers import TFSequenceClassifierOutput
                golden_out = golden_out.logits
            else:
                golden_out = golden_out.last_hidden_state
            # Save the artifacts in the directory dir.
            self.save_data(
                dir,
                model_name,
                imported_mlir[0],
                imported_mlir[1],
                self.inputs,
                golden_out,
            )
            return (
                imported_mlir,
                self.convert_to_numpy(self.inputs),
                golden_out,
            )
        if self.frontend in ["tflite", "tf-lite"]:
            # TODO(Chi): Validate it for tflite models.
            golden_out = self.module.invoke_tflite(self.inputs)
            self.save_data(
                dir,
                model_name,
                imported_mlir[0],
                imported_mlir[1],
                self.inputs,
                golden_out,
            )
            return (
                imported_mlir,
                self.inputs,
                golden_out,
            )
