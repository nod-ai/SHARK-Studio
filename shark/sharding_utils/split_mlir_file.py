import os
import numpy as np
import re


def _get_mlir_file_data(fname):
    # Returns 2 dictionaries with data on variables used in the mlir file
    # var_def_dict: maps varaibles to the line they are defined
    # line_varts_dict: maps each lined to the variables called on that line

    var_pattern = "%[0-9_A-Za-z]*"
    def_pattern = "^\s*%[0-9_A-Za-z]*\s*="

    f_ = open(fname)
    flines = f_.readlines()
    f_.close()

    var_def_dict, line_vars_dict = {}, {}

    i = 1
    for line in flines:
        line_vars_dict[i] = re.findall(var_pattern, line)

        var_definition = re.findall(def_pattern, line)

        if len(var_definition) > 1:
            print("ERROR: line {} defines more than one variable".format(i))
            print(line)

        elif len(var_definition) == 1:
            var_name = re.findall(var_pattern, var_definition[0])

            if len(var_name) != 1:
                print(
                    "ERROR: line {} defines more than one variable".format(i)
                )
                print(line)

            else:
                if var_name[0] not in var_def_dict.keys():
                    var_def_dict[var_name[0]] = [i]
                else:
                    var_def_dict[var_name[0]].append(i)
        i += 1

    return var_def_dict, line_vars_dict


def _find_function(function_name, lines):
    for line in lines:
        if re.search("func\s{1,}@" + function_name, line):
            return line


def _find_initial_input_vars(function_name, lines):
    result = {}
    for line in lines:
        if re.search("func\s{1,}@" + function_name, line):
            init_inputs = re.findall(
                "%[0-9_A-Za-z]*\s*:\s*tensor<[a-z0-9]*>", line
            )
            for init_input in init_inputs:
                init_input = init_input.split(":")
                init_input = [x.strip() for x in init_input]
                result[init_input[0]] = init_input[1]
    return result


def _find_initial_output_vars(lines):
    return_line = ""
    tensor = ""
    for line in lines:
        if "return" in line:
            return_line = line
            tensors = re.findall("tensor<[0-9a-z?]*>", line)
            tensor = tensors[0]
    return return_line, tensor


def _get_tensor_dimensions(var_name, vdd, lines):
    line = lines[vdd[var_name][0] - 1]
    line = re.sub("\s", "", line)
    line = re.sub(r"[(){}]", "", line)
    dim_pattern = "tensor<\w*>$"
    result = re.search(dim_pattern, line)
    try:
        return result.group(0)
    except:
        return None


def _create_header_footer(
    function_name,
    inputs,
    outputs,
    vdd,
    lines,
    init_vars,
    first=False,
    last=False,
):

    # Create the header and footer for the generated function

    HEADER_STRING = "func.func @{}({}) -> ({}) {{"
    FOOTER_STRING = "\treturn {} : {}\n}}"

    arg_map = {}

    header_inputs, header_output_types, footer_outputs, footer_output_types = (
        "",
        "",
        "",
        "",
    )

    if not first:
        for x, i in zip(inputs, range(len(inputs))):
            if x in init_vars.keys():
                x_dim = init_vars[x]
            else:
                x_dim = _get_tensor_dimensions(x, vdd, lines)
            header_inputs += r"%generated_arg" + str(i) + ": " + x_dim
            if i != len(inputs) - 1:
                header_inputs += ", "
    else:
        line = _find_function(function_name, lines)
        line = re.sub("\s", "", line)
        pattern = "\([^()]*\)(?=->)"
        result = re.search(pattern, line)
        result = result.group(0)
        result = re.sub("[()]", "", result)
        header_inputs = result

    if not last:
        for x, i in zip(outputs, range(len(outputs))):
            if x in init_vars.keys():
                x_dim = init_vars[x]
            else:
                x_dim = _get_tensor_dimensions(x, vdd, lines)
            if x_dim is None:
                print(x)
            else:
                header_output_types += x_dim
                footer_outputs += x
                footer_output_types += x_dim
                if i != len(outputs) - 1:
                    header_output_types += ", "
                    footer_outputs += ", "
                    footer_output_types += ", "
        FOOTER_STRING = FOOTER_STRING.format(
            footer_outputs, footer_output_types
        )

    else:
        # for line in lines:
        # 	if 'return' in line:
        # 		FOOTER_STRING = line
        header_output_types += _find_initial_output_vars(lines)[1]
        FOOTER_STRING = "}"

    return (
        HEADER_STRING.format(
            function_name, header_inputs, header_output_types
        ),
        FOOTER_STRING,
    )


def _get_input_var_list(lower, upper, lvd, vdd, lines, init_vars):

    # returns a list of variables that need to be added as input, or included as constants
    # items in the list contain a tuple of the var name, and the line they are defined

    var_list = []
    seen = set()

    for i in range(lower, upper):
        for x in lvd[i]:
            try:
                d_line = vdd[x]
                in_range = False
                for y in d_line:
                    if y > lower and y < upper:
                        in_range = True
                if not in_range and x not in seen:
                    var_list.append((x, d_line))
                    seen.add(x)
            except KeyError:
                if x in init_vars.keys():
                    var_list.append((x, -1))

    return var_list


def _find_safe_splits(lines, function_name):
    in_clause = False
    in_function = False
    safe_splits = []
    bracket_debt = 0
    for line, i in zip(lines, range(len(lines))):
        if f"%0" in line:
            in_function = True
        if in_function:
            if not in_clause:
                safe_splits.append(i)
            open_brackets = line.count("{")
            close_brackets = line.count("}")
            bracket_diff = open_brackets - close_brackets
            bracket_debt += bracket_diff
            if bracket_debt == 0:
                in_clause = False
            elif bracket_debt != 0:
                in_clause = True

    return safe_splits


def _get_output_var_list(lower, upper, lvd, vdd, lines, init_vars):

    # returns list of variables that need to be outputed
    # items in the list contain a tuple of the var name, and the line they are defined

    var_list = []
    seen = set()

    for i in range(upper, len(lines)):
        for x in lvd[i]:
            try:
                d_line = vdd[x]
                in_range = True
                for y in d_line:
                    if not (y < upper):
                        in_range = False
                if (
                    in_range
                    and x not in seen
                    and "c" not in x
                    and "t" not in x
                ):
                    var_list.append((x, d_line))
                    seen.add(x)
            except KeyError:
                if x in init_vars.keys():
                    var_list.append((x, -1))

    return var_list


def _extract_in_out_cons(
    input_var_list, output_var_list, lower, upper, vdd, lines
):
    inputs, outputs, constants = [], [], []

    for x in input_var_list:
        if "c" in x[0] or "t" in x[0]:
            constants.append(lines[vdd[x[0]][0] - 1])
        elif x[1] == -1:
            inputs.append(x[0])

        else:
            if max(vdd[x[0]]) < upper:
                inputs.append(x[0])

    for x in output_var_list:
        outputs.append(x[0])

        if x[0] not in inputs and (x[1] == -1 or max(vdd[x[0]]) < lower):
            inputs.append(x[0])

    return inputs, outputs, constants


def _sub_variables_with_inputs(inputs, function_body, footer):
    for x, i in zip(inputs, range(len(inputs))):
        function_body = re.sub(
            f"{x}(?=\D)", "%generated_arg" + str(i), function_body
        )
        footer = re.sub(f"{x}(?=\D)", "%generated_arg" + str(i), footer)

    return function_body, footer


def split_mlir_file(
    fname, lower, upper, output_fname, first=False, last=False
):

    # split the lines of an mlir file into a subsection with relevant variables
    # currently no automatic adjustment if you try and cut in the middle of a clause, will be adding that next

    f_ = open(fname)
    flines = f_.readlines()
    f_.close()

    init_vars = _find_initial_input_vars("forward", flines)
    safe_splits = _find_safe_splits(flines, "forward")

    o_lower, o_upper = lower, upper
    while not (upper in safe_splits):
        if upper > max(safe_splits):
            upper = max(safe_splits)
        else:
            upper += 1
    while not (lower in safe_splits):
        lower += 1
    if o_lower != lower:
        print(f"Lower Bound shifted from {o_lower} to {lower}")
    if o_upper != upper:
        print(f"Upper Bound shifted from {o_upper} to {upper}")

    vdd, lvd = _get_mlir_file_data(fname)
    input_var_list = _get_input_var_list(
        lower, upper, lvd, vdd, flines, init_vars
    )
    output_var_list = _get_output_var_list(
        lower, upper, lvd, vdd, flines, init_vars
    )
    function_body = flines[lower:upper]
    inputs, outputs, constants = _extract_in_out_cons(
        input_var_list, output_var_list, lower, upper, vdd, flines
    )
    header, footer = _create_header_footer(
        "forward",
        inputs,
        outputs,
        vdd,
        flines,
        init_vars,
        first=first,
        last=last,
    )
    function_body = "".join(function_body)
    constants = "".join(constants)
    if not first:
        function_body, footer = _sub_variables_with_inputs(
            inputs, function_body, footer
        )

    f_ = open(output_fname, "w+")
    f_.write(header)
    f_.write("\n")
    f_.write(constants)
    f_.write(function_body)
    f_.write("\n")
    f_.write(footer)
    f_.close()
