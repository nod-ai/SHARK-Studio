import os
import tempfile
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx


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


def shark_compile_through_fx(
    model,
    inputs,
    extended_model_name,
    is_f16=False,
    f16_input_mask=None,
    save_dir=tempfile.gettempdir(),
    debug=False,
    generate_or_load_vmfb=True,
    extra_args=[],
    device=None,
    mlir_dialect="tm_tensor",
):
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
