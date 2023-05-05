import torch
import torch_mlir
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from typing import List
from io import BytesIO
from pathlib import Path
from shark.shark_downloader import download_public_file
from shark.shark_importer import transform_fx as transform_fx_
import re


def get_tank_vicuna_mlir(num):
    # name can be 1 or 2 for first and second vicuna model
    mname = {1: "FirstVicuna", 2: "SecondVicuna"}
    tank_url = "gs://shark_tank/FastChat/"
    download_public_file(tank_url, mname[num])
    print(f"Downloaded model : {mname[num]} from tank")


def get_torch_mlir_module_bytecode(model, model_inputs):
    fx_g = make_fx(
        model,
        decomposition_table=get_decompositions(
            [
                torch.ops.aten.embedding_dense_backward,
                torch.ops.aten.native_layer_norm_backward,
                torch.ops.aten.slice_backward,
                torch.ops.aten.select_backward,
                torch.ops.aten.norm.ScalarOpt_dim,
                torch.ops.aten.native_group_norm,
                torch.ops.aten.upsample_bilinear2d.vec,
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,
            ]
        ),
    )(*model_inputs)

    print("Got FX_G")

    def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
        removed_indexes = []
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, (list, tuple)):
                    node_arg = list(node_arg)
                    node_args_len = len(node_arg)
                    for i in range(node_args_len):
                        curr_index = node_args_len - (i + 1)
                        if node_arg[curr_index] is None:
                            removed_indexes.append(curr_index)
                            node_arg.pop(curr_index)
                    node.args = (tuple(node_arg),)
                    break

        if len(removed_indexes) > 0:
            fx_g.graph.lint()
            fx_g.graph.eliminate_dead_code()
            fx_g.recompile()
        removed_indexes.sort()
        return removed_indexes

    def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
        """
        Replace tuple with tuple element in functions that return one-element tuples.
        Returns true if an unwrapping took place, and false otherwise.
        """
        unwrapped_tuple = False
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, tuple):
                    if len(node_arg) == 1:
                        node.args = (node_arg[0],)
                        unwrapped_tuple = True
                        break

        if unwrapped_tuple:
            fx_g.graph.lint()
            fx_g.recompile()
        return unwrapped_tuple

    def transform_fx(fx_g):
        for node in fx_g.graph.nodes:
            if node.op == "call_function":
                if node.target in [
                    torch.ops.aten.empty,
                ]:
                    # aten.empty should be filled with zeros.
                    if node.target in [torch.ops.aten.empty]:
                        with fx_g.graph.inserting_after(node):
                            new_node = fx_g.graph.call_function(
                                torch.ops.aten.zero_,
                                args=(node,),
                            )
                            node.append(new_node)
                            node.replace_all_uses_with(new_node)
                            new_node.args = (node,)

        fx_g.graph.lint()

    transform_fx(fx_g)
    fx_g.recompile()
    removed_none_indexes = _remove_nones(fx_g)
    was_unwrapped = _unwrap_single_tuple_return(fx_g)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    print("FX_G recompile")

    def strip_overloads(gm):
        """
        Modifies the target of graph nodes in :attr:`gm` to strip overloads.
        Args:
            gm(fx.GraphModule): The input Fx graph module to be modified
        """
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                node.target = node.target.overloadpacket
        gm.recompile()

    strip_overloads(fx_g)
    ts_g = torch.jit.script(fx_g)
    print("Got TS_G")

    return ts_g


def compile_vicuna(model, model_inputs, model_name, model_vmfb_name):
    # ADD Device Arg
    from shark.shark_inference import SharkInference

    vmfb_path = Path(model_vmfb_name + ".vmfb")
    if vmfb_path.exists():
        shark_module = SharkInference(
            None, device="cuda", mlir_dialect="tm_tensor"
        )
        shark_module.load_module(vmfb_path)
        return shark_module

    mlir_path = Path(model_name + ".mlir")
    print(
        f"[DEBUG] mlir path { mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
    )
    if mlir_path.exists():
        with open(mlir_path, "rb") as f:
            bytecode = f.read()
    else:
        ts_graph = get_torch_mlir_module_bytecode(model, model_inputs)
        # model_inputs = list(model_inputs)
        # model_inputs[0] = torch_mlir.TensorPlaceholder.like(model_inputs[0], dynamic_axes=[1])
        # model_inputs = tuple(model_inputs)
        module = torch_mlir.compile(
            ts_graph,
            [*model_inputs],
            torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=False,
            verbose=False,
        )

        def remove_constant_dim(line):
            if "19x" in line:
                line = re.sub("19x", "?x", line)
                line = re.sub("tensor.empty\(\)", "tensor.empty(%dim)", line)
            if "tensor.empty" in line and "?x?" in line:
                line = re.sub(
                    "tensor.empty\(%dim\)", "tensor.empty(%dim, %dim)", line
                )
            if "arith.cmpi" in line:
                line = re.sub("c19", "dim", line)
            if " 19," in line:
                line = re.sub(" 19,", " %dim,", line)
            return line

        bytecode_stream = BytesIO()
        module.operation.write_bytecode(bytecode_stream)
        bytecode = bytecode_stream.getvalue()
    f_ = open(model_name + ".mlir", "wb")
    f_.write(bytecode)
    print("Saved mlir")
    f_.close()

    shark_module = SharkInference(
        mlir_module=bytecode, device="cuda", mlir_dialect="tm_tensor"
    )
    # shark_module.compile()

    import os

    path = shark_module.save_module(os.getcwd(), model_vmfb_name, [])
    print("Saved vmfb at ", str(path))

    return shark_module


kwargs = {"torch_dtype": torch.float32}  # 16
model_path = "TheBloke/vicuna-7B-1.1-HF"


# Requires input_ids as tensor(1x40)
class FirstVicuna(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )  # .cuda().half()

    def forward(self, input_ids, attention_mask):
        # input_len = input_id_len
        # input_ids = input_ids[:,:input_len].reshape([1,input_len])
        op = self.model(
            input_ids=input_ids, use_cache=True, attention_mask=attention_mask
        )
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


# Uncomment this after verifying that SecondVicuna compiles as well.
# Might have to cast to_numpy.


# Requires input_ids as tensor(1x1),
#          past_key_values = 32 length tuple containing tuple of tensor pairs, which is same as output
#                            of firstVicuna[1:]
class SecondVicuna_(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

    def forward(self, input_tuple):
        # input_ids = input_tuple[0]
        # input_tuple = torch.unbind(pkv, dim=0)
        past_key_values = [
            (
                input_tuple[i],
                input_tuple[i + 1],
            )
            for i in range(0, len(input_tuple) - 1, 2)
        ]
        # for e1, e2 in zip(input_tuple, input_tuple[1:]):
        #    past_key_values.append(tuple(e1, e2))
        past_key_values = tuple(past_key_values)
        op = self.model(
            input_ids=token, use_cache=True, past_key_values=past_key_values
        )
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class SecondVicuna(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )  # .cuda().half()

    def forward(
        self,
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,
        i7,
        i8,
        i9,
        i10,
        i11,
        i12,
        i13,
        i14,
        i15,
        i16,
        i17,
        i18,
        i19,
        i20,
        i21,
        i22,
        i23,
        i24,
        i25,
        i26,
        i27,
        i28,
        i29,
        i30,
        i31,
        i32,
        i33,
        i34,
        i35,
        i36,
        i37,
        i38,
        i39,
        i40,
        i41,
        i42,
        i43,
        i44,
        i45,
        i46,
        i47,
        i48,
        i49,
        i50,
        i51,
        i52,
        i53,
        i54,
        i55,
        i56,
        i57,
        i58,
        i59,
        i60,
        i61,
        i62,
        i63,
        i64,
    ):
        # input_ids = input_tuple[0]
        # input_tuple = torch.unbind(pkv, dim=0)
        token = i0
        past_key_values = (
            (i1, i2),
            (
                i3,
                i4,
            ),
            (
                i5,
                i6,
            ),
            (
                i7,
                i8,
            ),
            (
                i9,
                i10,
            ),
            (
                i11,
                i12,
            ),
            (
                i13,
                i14,
            ),
            (
                i15,
                i16,
            ),
            (
                i17,
                i18,
            ),
            (
                i19,
                i20,
            ),
            (
                i21,
                i22,
            ),
            (
                i23,
                i24,
            ),
            (
                i25,
                i26,
            ),
            (
                i27,
                i28,
            ),
            (
                i29,
                i30,
            ),
            (
                i31,
                i32,
            ),
            (
                i33,
                i34,
            ),
            (
                i35,
                i36,
            ),
            (
                i37,
                i38,
            ),
            (
                i39,
                i40,
            ),
            (
                i41,
                i42,
            ),
            (
                i43,
                i44,
            ),
            (
                i45,
                i46,
            ),
            (
                i47,
                i48,
            ),
            (
                i49,
                i50,
            ),
            (
                i51,
                i52,
            ),
            (
                i53,
                i54,
            ),
            (
                i55,
                i56,
            ),
            (
                i57,
                i58,
            ),
            (
                i59,
                i60,
            ),
            (
                i61,
                i62,
            ),
            (
                i63,
                i64,
            ),
        )
        # for e1, e2 in zip(input_tuple, input_tuple[1:]):
        #    past_key_values.append(tuple(e1, e2))
        op = self.model(
            input_ids=token, use_cache=True, past_key_values=past_key_values
        )
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        pkv = [
            torch.rand([1, 32, 40, 128], dtype=torch.float32)
            for _ in range(64)
        ]
        return self.model(input_ids, past_key_values=pkv)


if __name__ == "__main__":
    import sys

    vicuna_number = 1

    # input_tuple = (torch.ones([1,1], dtype=torch.int),) + tuple(torch.rand([1, 32, 40, 128], dtype=torch.float32) for _ in range(64))
    # input_tuple = torch.rand([1,2])
    # secondVicuna = SecondVicuna(model_path)
    # shark_second_vicuna = compile_vicuna(secondVicuna, (input_tuple,), "second_vicuna.mlir", "second_vicuna")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # prompt = "INPUT: The SQL command to extract all the users whose name starts with A is:"
    prompt = "".join(["0" for _ in range(254)])
    input_ids = tokenizer(prompt).input_ids
    # print("Got input_ids from the tokenizer")

    if vicuna_number == 1:
        prompt = input("Enter Prompt: ")
        prompt = prompt.strip()
        input_ids = tokenizer(prompt).input_ids
        original_input_ids = input_ids
        input_id_len = len(input_ids)
        pad_len = 256 - input_id_len
        attention_mask = torch.ones([1, input_id_len], dtype=torch.int64)
        input_ids = torch.nn.functional.pad(
            torch.tensor(input_ids), (0, pad_len), mode="constant", value=259
        )
        input_ids = input_ids.reshape([1, 256])
        attention_mask = torch.nn.functional.pad(
            torch.tensor(attention_mask),
            (0, pad_len),
            mode="constant",
            value=0,
        )

        firstVicuna = FirstVicuna(model_path)

        prompt2 = "".join(["0" for _ in range(254)])
        input_ids2 = tokenizer(prompt2).input_ids
        input_ids2 = torch.tensor(input_ids2).reshape([1, 256])
        # firstVicunaInput = tuple([torch.as_tensor([input_ids])])#.cuda()
        # firstVicunaCompileInput = (input_ids2, torch.tensor([input_id_len]))
        firstVicunaCompileInput = (input_ids2, attention_mask)
        len_ = int(torch.tensor([input_id_len]))
        # firstVicunaInput = (input_ids,int(torch.tensor([input_id_len])), )
        firstVicunaInput = (
            input_ids,
            attention_mask,
        )

        shark_first_vicuna = compile_vicuna(
            firstVicuna,
            firstVicunaCompileInput,
            "first_vicuna",
            "first_vicuna",
        )
        # input_ids = torch.tensor(input_ids)

        # output_first_vicuna = shark_first_vicuna("forward", (input_ids.reshape([1, input_ids.shape[0]]),))
        output_first_vicuna = shark_first_vicuna("forward", firstVicunaInput)
        output_first_vicuna_tensor = torch.tensor(output_first_vicuna[1:])
        torch.save(output_first_vicuna_tensor, "outpt_first_vicuna_tensor.pt")
        logits_first_vicuna = torch.tensor(output_first_vicuna[0])
        torch.save(logits_first_vicuna, "logits_first_vicuna_tensor.pt")
        # output_non_shark_first_vicuna = firstVicuna.forward(firstVicunaInput[0])

        for i in range(40):
            original_input_ids.append(
                torch.argmax(logits_first_vicuna[:, len_ + i - 1, :], dim=1)
            )
            print(
                torch.argmax(logits_first_vicuna[:, len_ + i - 1, :], dim=1),
                tokenizer.decode(
                    torch.argmax(
                        logits_first_vicuna[:, len_ + i - 1, :], dim=1
                    )
                ),
            )
            input_id_len = len(original_input_ids)
            pad_len = 256 - input_id_len
            attention_mask = torch.ones([1, input_id_len], dtype=torch.int64)
            input_ids = torch.nn.functional.pad(
                torch.tensor(original_input_ids),
                (0, pad_len),
                mode="constant",
                value=259,
            )
            input_ids = input_ids.reshape([1, 256])
            attention_mask = torch.nn.functional.pad(
                torch.tensor(attention_mask),
                (0, pad_len),
                mode="constant",
                value=0,
            )
            firstVicunaInput = (
                input_ids,
                attention_mask,
            )
            output_first_vicuna = shark_first_vicuna(
                "forward", firstVicunaInput
            )
            output_first_vicuna_tensor = torch.tensor(output_first_vicuna[1:])
            logits_first_vicuna = torch.tensor(output_first_vicuna[0])

        print(
            tokenizer.decode(
                torch.argmax(logits_first_vicuna[:, len_ - 1, :], dim=1)
            )
        )

    if vicuna_number == 2:
        # last_token_logits = output_first_vicuna[0][0][-1]
        # print("SHARK firstVicuna = ", str(last_token_logits))
        # print("NonSHARK firstVicuna = ", str(output_non_shark_first_vicuna[0][0][-1]))

        # temperature = 0.7
        # probs = torch.softmax(torch.tensor(last_token_logits / temperature, dim=-1))
        # token = torch.tensor(int(torch.multinomial(probs, num_samples=1))).reshape([1,1])
        # token = torch.ones([1,1], dtype=torch.int64)#.cuda()
        # pkvt = []
        # for i in range(64):
        #    pkvt.append(torch.randn(1, 32, 40, 128, dtype=torch.float32))
        # pkvt = tuple(pkvt)

        # token = torch.ones([1,1], dtype=torch.int64)#.cuda()
        output_first_vicuna = torch.load("outpt_first_vicuna_tensor.pt")
        logits_first_vicuna = torch.load("logits_first_vicuna_tensor.pt")
        print(logits_first_vicuna.shape)

        for i in range(logits_first_vicuna.shape[1]):
            token = torch.argmax(
                torch.tensor(logits_first_vicuna)[:, i, :], dim=1
            ).reshape([1, 1])
            print(token, tokenizer.decode(token[0][0]))

        token = torch.argmax(
            torch.tensor(logits_first_vicuna)[:, 8, :], dim=1
        ).reshape([1, 1])
        print(logits_first_vicuna)
        print(torch.tensor(logits_first_vicuna)[:, -1, :])
        print(token, tokenizer.decode(token[0][0]))

        result = [tokenizer.decode(token[0][0])]

        pkvt = tuple(torch.tensor(x) for x in output_first_vicuna)
        # pkv = torch.stack(pkvt, dim=0)
        secondVicuna = SecondVicuna(model_path)
        # del shark_first_vicuna
        # del output_first_vicuna
        # torch.cuda.empty_cache()
        shark_second_vicuna = compile_vicuna(
            secondVicuna, (token,) + pkvt, "second_vicuna", "second_vicuna"
        )

        print(len(pkvt))

        output_second_vicuna = shark_second_vicuna("forward", (token,) + pkvt)

        import time

        f_ = open("all-logit-outputs.txt", "w+")

        print(output_second_vicuna[0].shape)

        for _ in range(10):
            f_.write(
                f"{_}:------------------------------------------------------------------------\n"
            )
            t1 = time.time()
            start_point = output_second_vicuna[1].shape[2] - 256
            for j in range(output_second_vicuna[0].shape[1]):
                token_test = torch.argmax(
                    torch.tensor(output_second_vicuna[0])[:, j, :], dim=1
                ).reshape([1, 1])
                sym = token_test, tokenizer.decode(token_test[0][0])
                f_.write(f"{i}: {token_test} | {sym}")
            token = torch.argmax(
                torch.tensor(output_second_vicuna[0])[:, -1, :], dim=1
            ).reshape([1, 1])
            # print(token, tokenizer.decode(token[0][0]))
            result.append(tokenizer.decode(token[0][0]))
            truncated_outputs = tuple(
                x[:, :, :256, :] for x in output_second_vicuna[1:]
            )
            output_second_vicuna = shark_second_vicuna(
                "forward", (token,) + truncated_outputs
            )
            # print(f"Token Generated in {time.time() - t1} seconds")
            f_.write("\n")

        f_.close()

        print(result)
