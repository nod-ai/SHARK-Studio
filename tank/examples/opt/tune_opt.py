import os
from pathlib import Path
from shark_tuner.codegen_tuner import SharkCodegenTuner
from shark_tuner.iree_utils import (
    dump_dispatches,
    create_context,
    export_module_to_mlir_file,
)
from shark_tuner.model_annotation import model_annotation
from shark_opt_wrapper import OPTForCausalLMModel
from transformers import AutoTokenizer, OPTForCausalLM
from shark.shark_importer import import_with_fx

NUM_ITERS = 400
MODEL_NAME = "facebook/opt-1.3b"
MODEL_FNAME = "opt-1_3b-causallm"

def load_mlir_module():
    hf_model = OPTForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    opt_model = OPTForCausalLMModel(hf_model)

    prompt = "What is the meaning of life?"
    model_inputs = tokenizer(prompt, return_tensors="pt")
    inputs = (
        model_inputs["input_ids"],
        model_inputs["attention_mask"],
    )
    
    (
        mlir_module,
        func_name,
    ) = import_with_fx(
        model=opt_model,
        inputs=inputs,
        is_f16=False,
        model_name=MODEL_NAME.split("/")[1],
    )
    return mlir_module, model_name


def main():
    #mlir_module, model_name = load_mlir_module()

    # Get device and device specific arguments
    device = "cpu"

    # Dump model dispatches
    model_name = MODEL_NAME
    #generates_dir = "."
    #if not os.path.exists(generates_dir):
    #    os.makedirs(generates_dir)
    #dump_mlir = generates_dir / "temp.mlir"
    dispatch_dir = f"./{MODEL_FNAME}_{device}_dispatches"
    #export_module_to_mlir_file(mlir_module, dump_mlir)
    #dump_dispatches(
    #    dump_mlir,
    #    device,
    #    dispatch_dir,
    #)

    # Tune each dispatch
    dtype = "f32"
    config_filename = f"{MODEL_FNAME}_{device}_configs.json"
    for f_path in os.listdir(dispatch_dir):
        if not f_path.endswith(".mlir"):
            continue

        model_dir = os.path.join(dispatch_dir, f_path)

        tuner = SharkCodegenTuner(
            model_dir,
            device,
            "random",
            NUM_ITERS,
            ".",
            dtype,
            search_op="all",
            batch_size=1,
            config_filename=config_filename,
            use_dispatch=True,
        )
        tuner.tune()


if __name__ == "__main__":
    main()
