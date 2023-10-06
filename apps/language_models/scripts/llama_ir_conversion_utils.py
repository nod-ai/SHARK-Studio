from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter
import re, gc

"""
    This script can be used as a standalone utility to convert IRs to dynamic + combine them.
    Following are the various ways this script can be used :-
        a. To convert a single Linalg IR to dynamic IR:
            --dynamic --first_ir_path=<PATH TO FIRST IR>
        b. To convert two Linalg IRs to dynamic IR:
            --dynamic --first_ir_path=<PATH TO SECOND IR> --first_ir_path=<PATH TO SECOND IR>
        c. To combine two Linalg IRs into one:
            --combine --first_ir_path=<PATH TO FIRST IR> --second_ir_path=<PATH TO SECOND IR>
        d. To convert both IRs into dynamic as well as combine the IRs:
            --dynamic --combine --first_ir_path=<PATH TO FIRST IR> --second_ir_path=<PATH TO SECOND IR>

    NOTE: For dynamic you'll also need to provide the following set of flags:-
           i. For First Llama : --dynamic_input_size (DEFAULT: 19)
          ii. For Second Llama: --model_name (DEFAULT: llama2_7b)
                                --precision (DEFAULT: 'int4')
          You may use --save_dynamic to also save the dynamic IR in option d above.
          Else for option a. and b. the dynamic IR(s) will get saved by default.
"""


def combine_mlir_scripts(
    first_vicuna_mlir,
    second_vicuna_mlir,
    output_name,
    return_ir=True,
):
    print(f"[DEBUG] combining first and second mlir")
    print(f"[DEBUG] output_name = {output_name}")
    maps1 = []
    maps2 = []
    constants = set()
    f1 = []
    f2 = []

    print(f"[DEBUG] processing first vicuna mlir")
    first_vicuna_mlir = first_vicuna_mlir.splitlines()
    while first_vicuna_mlir:
        line = first_vicuna_mlir.pop(0)
        if re.search("#map\d*\s*=", line):
            maps1.append(line)
        elif re.search("arith.constant", line):
            constants.add(line)
        elif not re.search("module", line):
            line = re.sub("forward", "first_vicuna_forward", line)
            f1.append(line)
    f1 = f1[:-1]
    del first_vicuna_mlir
    gc.collect()

    for i, map_line in enumerate(maps1):
        map_var = map_line.split(" ")[0]
        map_line = re.sub(f"{map_var}(?!\d)", map_var + "_0", map_line)
        maps1[i] = map_line
        f1 = [
            re.sub(f"{map_var}(?!\d)", map_var + "_0", func_line)
            for func_line in f1
        ]

    print(f"[DEBUG] processing second vicuna mlir")
    second_vicuna_mlir = second_vicuna_mlir.splitlines()
    while second_vicuna_mlir:
        line = second_vicuna_mlir.pop(0)
        if re.search("#map\d*\s*=", line):
            maps2.append(line)
        elif "global_seed" in line:
            continue
        elif re.search("arith.constant", line):
            constants.add(line)
        elif not re.search("module", line):
            line = re.sub("forward", "second_vicuna_forward", line)
            f2.append(line)
    f2 = f2[:-1]
    del second_vicuna_mlir
    gc.collect()

    for i, map_line in enumerate(maps2):
        map_var = map_line.split(" ")[0]
        map_line = re.sub(f"{map_var}(?!\d)", map_var + "_1", map_line)
        maps2[i] = map_line
        f2 = [
            re.sub(f"{map_var}(?!\d)", map_var + "_1", func_line)
            for func_line in f2
        ]

    module_start = 'module attributes {torch.debug_module_name = "_lambda"} {'
    module_end = "}"

    global_vars = []
    vnames = []
    global_var_loading1 = []
    global_var_loading2 = []

    print(f"[DEBUG] processing constants")
    counter = 0
    constants = list(constants)
    while constants:
        constant = constants.pop(0)
        vname, vbody = constant.split("=")
        vname = re.sub("%", "", vname)
        vname = vname.strip()
        vbody = re.sub("arith.constant", "", vbody)
        vbody = vbody.strip()
        if len(vbody.split(":")) < 2:
            print(constant)
        vdtype = vbody.split(":")[-1].strip()
        fixed_vdtype = vdtype
        if "c1_i64" in vname:
            print(constant)
            counter += 1
        if counter == 2:
            counter = 0
            print("detected duplicate")
            continue
        vnames.append(vname)
        if "true" not in vname:
            global_vars.append(
                f"ml_program.global private @{vname}({vbody}) : {fixed_vdtype}"
            )
            global_var_loading1.append(
                f"\t\t%{vname} = ml_program.global_load_const @{vname} : {fixed_vdtype}"
            )
            global_var_loading2.append(
                f"\t\t%{vname} = ml_program.global_load_const @{vname} : {fixed_vdtype}"
            )
        else:
            global_vars.append(
                f"ml_program.global private @{vname}({vbody}) : i1"
            )
            global_var_loading1.append(
                f"\t\t%{vname} = ml_program.global_load_const @{vname} : i1"
            )
            global_var_loading2.append(
                f"\t\t%{vname} = ml_program.global_load_const @{vname} : i1"
            )

    new_f1, new_f2 = [], []

    print(f"[DEBUG] processing f1")
    for line in f1:
        if "func.func" in line:
            new_f1.append(line)
            for global_var in global_var_loading1:
                new_f1.append(global_var)
        else:
            new_f1.append(line)

    print(f"[DEBUG] processing f2")
    for line in f2:
        if "func.func" in line:
            new_f2.append(line)
            for global_var in global_var_loading2:
                if (
                    "c20_i64 = arith.addi %dim_i64, %c1_i64 : i64"
                    in global_var
                ):
                    print(global_var)
                new_f2.append(global_var)
        else:
            new_f2.append(line)

    f1 = new_f1
    f2 = new_f2

    del new_f1
    del new_f2
    gc.collect()

    print(
        [
            "c20_i64 = arith.addi %dim_i64, %c1_i64 : i64" in x
            for x in [maps1, maps2, global_vars, f1, f2]
        ]
    )

    # doing it this way rather than assembling the whole string
    # to prevent OOM with 64GiB RAM when encoding the file.

    print(f"[DEBUG] Saving mlir to {output_name}")
    with open(output_name, "w+") as f_:
        f_.writelines(line + "\n" for line in maps1)
        f_.writelines(line + "\n" for line in maps2)
        f_.writelines(line + "\n" for line in [module_start])
        f_.writelines(line + "\n" for line in global_vars)
        f_.writelines(line + "\n" for line in f1)
        f_.writelines(line + "\n" for line in f2)
        f_.writelines(line + "\n" for line in [module_end])

    del maps1
    del maps2
    del module_start
    del global_vars
    del f1
    del f2
    del module_end
    gc.collect()

    if return_ir:
        print(f"[DEBUG] Reading combined mlir back in")
        with open(output_name, "rb") as f:
            return f.read()


def write_in_dynamic_inputs0(module, dynamic_input_size):
    print("[DEBUG] writing dynamic inputs to first vicuna")
    # Current solution for ensuring mlir files support dynamic inputs
    # TODO: find a more elegant way to implement this
    new_lines = []
    module = module.splitlines()
    while module:
        line = module.pop(0)
        line = re.sub(f"{dynamic_input_size}x", "?x", line)
        if "?x" in line:
            line = re.sub("tensor.empty\(\)", "tensor.empty(%dim)", line)
        line = re.sub(f" {dynamic_input_size},", " %dim,", line)
        if "tensor.empty" in line and "?x?" in line:
            line = re.sub(
                "tensor.empty\(%dim\)", "tensor.empty(%dim, %dim)", line
            )
        if "arith.cmpi" in line:
            line = re.sub(f"c{dynamic_input_size}", "dim", line)
        if "%0 = tensor.empty(%dim) : tensor<?xi64>" in line:
            new_lines.append("%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>")
        if "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>" in line:
            continue

        new_lines.append(line)
    return "\n".join(new_lines)


def write_in_dynamic_inputs1(module, model_name, precision):
    print("[DEBUG] writing dynamic inputs to second vicuna")

    def remove_constant_dim(line):
        if "c19_i64" in line:
            line = re.sub("c19_i64", "dim_i64", line)
        if "19x" in line:
            line = re.sub("19x", "?x", line)
            line = re.sub("tensor.empty\(\)", "tensor.empty(%dim)", line)
        if "tensor.empty" in line and "?x?" in line:
            line = re.sub(
                "tensor.empty\(%dim\)",
                "tensor.empty(%dim, %dim)",
                line,
            )
        if "arith.cmpi" in line:
            line = re.sub("c19", "dim", line)
        if " 19," in line:
            line = re.sub(" 19,", " %dim,", line)
        if "x20x" in line or "<20x" in line:
            line = re.sub("20x", "?x", line)
            line = re.sub("tensor.empty\(\)", "tensor.empty(%dimp1)", line)
        if " 20," in line:
            line = re.sub(" 20,", " %dimp1,", line)
        return line

    module = module.splitlines()
    new_lines = []

    # Using a while loop and the pop method to avoid creating a copy of module
    if "llama2_13b" in model_name:
        pkv_tensor_shape = "tensor<1x40x?x128x"
    elif "llama2_70b" in model_name:
        pkv_tensor_shape = "tensor<1x8x?x128x"
    else:
        pkv_tensor_shape = "tensor<1x32x?x128x"
    if precision in ["fp16", "int4", "int8"]:
        pkv_tensor_shape += "f16>"
    else:
        pkv_tensor_shape += "f32>"

    while module:
        line = module.pop(0)
        if "%c19_i64 = arith.constant 19 : i64" in line:
            new_lines.append("%c2 = arith.constant 2 : index")
            new_lines.append(
                f"%dim_4_int = tensor.dim %arg1, %c2 : {pkv_tensor_shape}"
            )
            new_lines.append(
                "%dim_i64 = arith.index_cast %dim_4_int : index to i64"
            )
            continue
        if "%c2 = arith.constant 2 : index" in line:
            continue
        if "%c20_i64 = arith.constant 20 : i64" in line:
            new_lines.append("%c1_i64 = arith.constant 1 : i64")
            new_lines.append("%c20_i64 = arith.addi %dim_i64, %c1_i64 : i64")
            new_lines.append(
                "%dimp1 = arith.index_cast %c20_i64 : i64 to index"
            )
            continue
        line = remove_constant_dim(line)
        new_lines.append(line)

    return "\n".join(new_lines)


def save_dynamic_ir(ir_to_save, output_file):
    if not ir_to_save:
        return
    # We only get string output from the dynamic conversion utility.
    from contextlib import redirect_stdout

    with open(output_file, "w") as f:
        with redirect_stdout(f):
            print(ir_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="llama ir utility",
        description="\tThis script can be used as a standalone utility to convert IRs to dynamic + combine them.\n"
        + "\tFollowing are the various ways this script can be used :-\n"
        + "\t\ta. To convert a single Linalg IR to dynamic IR:\n"
        + "\t\t\t--dynamic --first_ir_path=<PATH TO FIRST IR>\n"
        + "\t\tb. To convert two Linalg IRs to dynamic IR:\n"
        + "\t\t\t--dynamic --first_ir_path=<PATH TO SECOND IR> --first_ir_path=<PATH TO SECOND IR>\n"
        + "\t\tc. To combine two Linalg IRs into one:\n"
        + "\t\t\t--combine --first_ir_path=<PATH TO FIRST IR> --second_ir_path=<PATH TO SECOND IR>\n"
        + "\t\td. To convert both IRs into dynamic as well as combine the IRs:\n"
        + "\t\t\t--dynamic --combine --first_ir_path=<PATH TO FIRST IR> --second_ir_path=<PATH TO SECOND IR>\n\n"
        + "\tNOTE: For dynamic you'll also need to provide the following set of flags:-\n"
        + "\t\t i. For First Llama : --dynamic_input_size (DEFAULT: 19)\n"
        + "\t\tii. For Second Llama: --model_name (DEFAULT: llama2_7b)\n"
        + "\t\t\t--precision (DEFAULT: 'int4')\n"
        + "\t      You may use --save_dynamic to also save the dynamic IR in option d above.\n"
        + "\t      Else for option a. and b. the dynamic IR(s) will get saved by default.\n",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--precision",
        "-p",
        default="int4",
        choices=["fp32", "fp16", "int8", "int4"],
        help="Precision of the concerned IR",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama2_7b",
        choices=["vicuna", "llama2_7b", "llama2_13b", "llama2_70b"],
        help="Specify which model to run.",
    )
    parser.add_argument(
        "--first_ir_path",
        default=None,
        help="path to first llama mlir file",
    )
    parser.add_argument(
        "--second_ir_path",
        default=None,
        help="path to second llama mlir file",
    )
    parser.add_argument(
        "--dynamic_input_size",
        type=int,
        default=19,
        help="Specify the static input size to replace with dynamic dim.",
    )
    parser.add_argument(
        "--dynamic",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Converts the IR(s) to dynamic",
    )
    parser.add_argument(
        "--save_dynamic",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Save the individual IR(s) after converting to dynamic",
    )
    parser.add_argument(
        "--combine",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Converts the IR(s) to dynamic",
    )

    args, unknown = parser.parse_known_args()

    dynamic = args.dynamic
    combine = args.combine
    assert (
        dynamic or combine
    ), "neither `dynamic` nor `combine` flag is turned on"
    first_ir_path = args.first_ir_path
    second_ir_path = args.second_ir_path
    assert first_ir_path or second_ir_path, "no input ir has been provided"
    if combine:
        assert (
            first_ir_path and second_ir_path
        ), "you will need to provide both IRs to combine"
    precision = args.precision
    model_name = args.model_name
    dynamic_input_size = args.dynamic_input_size
    save_dynamic = args.save_dynamic

    print(f"Dynamic conversion utility is turned {'ON' if dynamic else 'OFF'}")
    print(f"Combining IR utility is turned {'ON' if combine else 'OFF'}")

    if dynamic and not combine:
        save_dynamic = True

    first_ir = None
    first_dynamic_ir_name = None
    second_ir = None
    second_dynamic_ir_name = None
    if first_ir_path:
        first_dynamic_ir_name = f"{Path(first_ir_path).stem}_dynamic"
        with open(first_ir_path, "r") as f:
            first_ir = f.read()
    if second_ir_path:
        second_dynamic_ir_name = f"{Path(second_ir_path).stem}_dynamic"
        with open(second_ir_path, "r") as f:
            second_ir = f.read()
    if dynamic:
        first_ir = (
            write_in_dynamic_inputs0(first_ir, dynamic_input_size)
            if first_ir
            else None
        )
        second_ir = (
            write_in_dynamic_inputs1(second_ir, model_name, precision)
            if second_ir
            else None
        )
        if save_dynamic:
            save_dynamic_ir(first_ir, f"{first_dynamic_ir_name}.mlir")
            save_dynamic_ir(second_ir, f"{second_dynamic_ir_name}.mlir")

    if combine:
        combine_mlir_scripts(
            first_ir,
            second_ir,
            f"{model_name}_{precision}.mlir",
            return_ir=False,
        )
