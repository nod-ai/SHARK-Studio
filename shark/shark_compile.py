import os
import tempfile
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
import torch
import torch_mlir
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from typing import List, Tuple
from io import BytesIO
from brevitas_examples.llm.llm_quant.quantize import quantize_model
from brevitas_examples.llm.llm_quant.run_utils import get_model_impl

def brevitas〇matmul_rhs_group_quant〡shape(lhs: List[int], rhs: List[int], rhs_scale: List[int], rhs_zero_point: List[int], rhs_bit_width: int, rhs_group_size: int) -> List[int]:
    if len(lhs) == 3 and len(rhs) == 2:
        return [lhs[0], lhs[1], rhs[0]]
    elif len(lhs) == 2 and len(rhs) == 2:
        return [lhs[0], rhs[0]]
    else:
        raise ValueError("Input shapes not supported.")


def brevitas〇matmul_rhs_group_quant〡dtype(lhs_rank_dtype: Tuple[int, int], rhs_rank_dtype: Tuple[int, int], rhs_scale_rank_dtype: Tuple[int, int], rhs_zero_point_rank_dtype: Tuple[int, int], rhs_bit_width: int, rhs_group_size: int) -> int:
    # output dtype is the dtype of the lhs float input
    lhs_rank, lhs_dtype = lhs_rank_dtype
    return lhs_dtype


def brevitas〇matmul_rhs_group_quant〡has_value_semantics(lhs, rhs, rhs_scale, rhs_zero_point, rhs_bit_width, rhs_group_size) -> None:
    return


brevitas_matmul_rhs_group_quant_library = [
    brevitas〇matmul_rhs_group_quant〡shape,
    brevitas〇matmul_rhs_group_quant〡dtype,
    brevitas〇matmul_rhs_group_quant〡has_value_semantics]


def load_vmfb(extended_model_name, device, mlir_dialect, extra_args=[]):
    vmfb_path = os.path.join(os.getcwd(), extended_model_name + ".vmfb")
    shark_module = None
    if os.path.isfile(vmfb_path):
        shark_module = SharkInference(
            None,
            device=device,
            mlir_dialect=mlir_dialect,
        )
        print(f"loading existing vmfb from: {vmfb_path}")
        shark_module.load_module(vmfb_path, extra_args=extra_args)
    return shark_module


def compile_module(
    shark_module, extended_model_name, generate_vmfb, extra_args=[]
):
    if generate_vmfb:
        vmfb_path = os.path.join(os.getcwd(), extended_model_name + ".vmfb")
        if os.path.isfile(vmfb_path):
            print(f"loading existing vmfb from: {vmfb_path}")
            shark_module.load_module(vmfb_path, extra_args=extra_args)
        else:
            print(
                "No vmfb found. Compiling and saving to {}".format(vmfb_path)
            )
            path = shark_module.save_module(
                os.getcwd(), extended_model_name, extra_args
            )
            shark_module.load_module(path, extra_args=extra_args)
    else:
        shark_module.compile(extra_args)
    return shark_module


def compile_int_precision(model, inputs, precision, device, generate_vmfb, extended_model_name):
    weight_bit_width = 4 if precision == "int4" else 8
    weight_group_size = 128
    quantize_model(
        get_model_impl(model),
        dtype=torch.float32,
        weight_quant_type="asym",
        weight_bit_width=weight_bit_width,
        weight_param_method="stats",
        weight_scale_precision="float",
        weight_quant_granularity="per_group",
        weight_group_size=weight_group_size,
        quantize_weight_zero_point=False,
        input_bit_width=None,
        input_scale_type="float",
        input_param_method="stats",
        input_quant_type="asym",
        input_quant_granularity="per_tensor",
        quantize_input_zero_point=False,
        seqlen=2048,
    )
    print("Weight quantization applied.")
    torchscript_module = import_with_fx(
        model,
        inputs,
        precision=precision,
        mlir_type="torchscript",
    )
    mlir_module = torch_mlir.compile(
        torchscript_module,
        inputs,
        output_type="torch",
        backend_legal_ops=["brevitas.matmul_rhs_group_quant"],
        extra_library=brevitas_matmul_rhs_group_quant_library,
        use_tracing=False,
        verbose=False,
    )
    print(f"[DEBUG] converting torch to linalg")
    run_pipeline_with_repro_report(
        mlir_module,
        "builtin.module(func.func(torch-unpack-torch-tensor),torch-backend-to-linalg-on-tensors-backend-pipeline)",
        description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
    )
    from contextlib import redirect_stdout

    mlir_file_path = os.path.join(os.getcwd(), f"{extended_model_name}_linalg.mlir")
    with open(mlir_file_path, 'w') as f:
        with redirect_stdout(f):
            print(mlir_module.operation.get_asm())
    mlir_module = str(mlir_module)
    mlir_module = mlir_module.encode("UTF-8")
    mlir_module = BytesIO(mlir_module)
    bytecode = mlir_module.read()
    print(f"Elided IR written for {extended_model_name}")
    return bytecode
    shark_module = SharkInference(
        mlir_module=bytecode, device=device, mlir_dialect="tm_tensor"
    )
    extra_args = [
        "--iree-hal-dump-executable-sources-to=ies",
        "--iree-vm-target-truncate-unsupported-floats",
        "--iree-codegen-check-ir-before-llvm-conversion=false",
        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
    ]
    return (
        compile_module(shark_module, extended_model_name=extended_model_name, generate_vmfb=generate_vmfb, extra_args=extra_args),
        bytecode
    )

def shark_compile_through_fx(
    model,
    inputs,
    extended_model_name,
    precision,
    f16_input_mask=None,
    save_dir=tempfile.gettempdir(),
    debug=False,
    generate_or_load_vmfb=True,
    extra_args=[],
    device=None,
    mlir_dialect="tm_tensor",
):
    is_f16 = precision == "fp16"
    if generate_or_load_vmfb:
        shark_module = load_vmfb(
            extended_model_name=extended_model_name,
            device=device,
            mlir_dialect=mlir_dialect,
            extra_args=extra_args,
        )
        if shark_module:
            return (
                shark_module,
                None,
            )

    from shark.parser import shark_args

    if "cuda" in device:
        shark_args.enable_tf32 = True

    if precision in ["int4", "int8"]:
        mlir_module = compile_int_precision(model, inputs, precision, device, generate_or_load_vmfb, extended_model_name)
        extra_args = [
            "--iree-hal-dump-executable-sources-to=ies",
            "--iree-vm-target-truncate-unsupported-floats",
            "--iree-codegen-check-ir-before-llvm-conversion=false",
            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
        ]
    else:
        (
            mlir_module,
            _,
        ) = import_with_fx(
            model=model,
            inputs=inputs,
            is_f16=is_f16,
            f16_input_mask=f16_input_mask,
            debug=debug,
            model_name=extended_model_name,
            save_dir=save_dir,
        )

    shark_module = SharkInference(
        mlir_module,
        device=device,
        mlir_dialect=mlir_dialect,
    )
    return (
        compile_module(
            shark_module,
            extended_model_name,
            generate_vmfb=generate_or_load_vmfb,
            extra_args=extra_args,
        ),
        mlir_module,
    )
