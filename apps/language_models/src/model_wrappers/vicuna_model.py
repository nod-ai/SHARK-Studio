import torch
from transformers import AutoModelForCausalLM


class FirstVicuna(torch.nn.Module):
    def __init__(
        self,
        model_path,
        precision="fp32",
        accumulates="fp32",
        weight_group_size=128,
        model_name="vicuna",
        hf_auth_token: str = None,
    ):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        if "llama2" in model_name:
            kwargs["use_auth_token"] = hf_auth_token
        self.accumulates = (
            torch.float32 if accumulates == "fp32" else torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        print(f"[DEBUG] model_path : {model_path}")
        if precision in ["int4", "int8"]:
            from brevitas_examples.common.generative.quantize import (
                quantize_model,
            )
            from brevitas_examples.llm.llm_quant.run_utils import (
                get_model_impl,
            )

            print("First Vicuna applying weight quantization..")
            weight_bit_width = 4 if precision == "int4" else 8
            quantize_model(
                get_model_impl(self.model).layers,
                dtype=self.accumulates,
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float_scale",
                weight_quant_type="asym",
                weight_quant_granularity="per_group",
                weight_group_size=weight_group_size,
                quantize_weight_zero_point=False,
            )
            print("Weight quantization applied.")

    def forward(self, input_ids):
        op = self.model(input_ids=input_ids, use_cache=True)
        return_vals = []
        token = torch.argmax(op.logits[:, -1, :], dim=1)
        return_vals.append(token)

        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class SecondVicuna7B(torch.nn.Module):
    def __init__(
        self,
        model_path,
        precision="fp32",
        accumulates="fp32",
        weight_group_size=128,
        model_name="vicuna",
        hf_auth_token: str = None,
    ):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        if "llama2" in model_name:
            kwargs["use_auth_token"] = hf_auth_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        self.accumulates = (
            torch.float32 if accumulates == "fp32" else torch.float16
        )
        print(f"[DEBUG] model_path : {model_path}")
        if precision in ["int4", "int8"]:
            from brevitas_examples.common.generative.quantize import (
                quantize_model,
            )
            from brevitas_examples.llm.llm_quant.run_utils import (
                get_model_impl,
            )

            print("Second Vicuna applying weight quantization..")
            weight_bit_width = 4 if precision == "int4" else 8
            quantize_model(
                get_model_impl(self.model).layers,
                dtype=self.accumulates,
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float_scale",
                weight_quant_type="asym",
                weight_quant_granularity="per_group",
                weight_group_size=weight_group_size,
                quantize_weight_zero_point=False,
            )
            print("Weight quantization applied.")

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
        op = self.model(
            input_ids=token, use_cache=True, past_key_values=past_key_values
        )
        return_vals = []
        token = torch.argmax(op.logits[:, -1, :], dim=1)
        return_vals.append(token)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class SecondVicuna13B(torch.nn.Module):
    def __init__(
        self,
        model_path,
        precision="int8",
        accumulates="fp32",
        weight_group_size=128,
        model_name="vicuna",
        hf_auth_token: str = None,
    ):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        if "llama2" in model_name:
            kwargs["use_auth_token"] = hf_auth_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        self.accumulates = (
            torch.float32 if accumulates == "fp32" else torch.float16
        )
        if precision in ["int4", "int8"]:
            from brevitas_examples.common.generative.quantize import (
                quantize_model,
            )
            from brevitas_examples.llm.llm_quant.run_utils import (
                get_model_impl,
            )

            print("Second Vicuna applying weight quantization..")
            weight_bit_width = 4 if precision == "int4" else 8
            quantize_model(
                get_model_impl(self.model).layers,
                dtype=self.accumulates,
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float_scale",
                weight_quant_type="asym",
                weight_quant_granularity="per_group",
                weight_group_size=weight_group_size,
                quantize_weight_zero_point=False,
            )
            print("Weight quantization applied.")

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
        i65,
        i66,
        i67,
        i68,
        i69,
        i70,
        i71,
        i72,
        i73,
        i74,
        i75,
        i76,
        i77,
        i78,
        i79,
        i80,
    ):
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
            (
                i65,
                i66,
            ),
            (
                i67,
                i68,
            ),
            (
                i69,
                i70,
            ),
            (
                i71,
                i72,
            ),
            (
                i73,
                i74,
            ),
            (
                i75,
                i76,
            ),
            (
                i77,
                i78,
            ),
            (
                i79,
                i80,
            ),
        )
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


class SecondVicuna70B(torch.nn.Module):
    def __init__(
        self,
        model_path,
        precision="fp32",
        accumulates="fp32",
        weight_group_size=128,
        model_name="vicuna",
        hf_auth_token: str = None,
    ):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        if "llama2" in model_name:
            kwargs["use_auth_token"] = hf_auth_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        self.accumulates = (
            torch.float32 if accumulates == "fp32" else torch.float16
        )
        print(f"[DEBUG] model_path : {model_path}")
        if precision in ["int4", "int8"]:
            from brevitas_examples.common.generative.quantize import (
                quantize_model,
            )
            from brevitas_examples.llm.llm_quant.run_utils import (
                get_model_impl,
            )

            print("Second Vicuna applying weight quantization..")
            weight_bit_width = 4 if precision == "int4" else 8
            quantize_model(
                get_model_impl(self.model).layers,
                dtype=self.accumulates,
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float_scale",
                weight_quant_type="asym",
                weight_quant_granularity="per_group",
                weight_group_size=weight_group_size,
                quantize_weight_zero_point=False,
            )
            print("Weight quantization applied.")

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
        i65,
        i66,
        i67,
        i68,
        i69,
        i70,
        i71,
        i72,
        i73,
        i74,
        i75,
        i76,
        i77,
        i78,
        i79,
        i80,
        i81,
        i82,
        i83,
        i84,
        i85,
        i86,
        i87,
        i88,
        i89,
        i90,
        i91,
        i92,
        i93,
        i94,
        i95,
        i96,
        i97,
        i98,
        i99,
        i100,
        i101,
        i102,
        i103,
        i104,
        i105,
        i106,
        i107,
        i108,
        i109,
        i110,
        i111,
        i112,
        i113,
        i114,
        i115,
        i116,
        i117,
        i118,
        i119,
        i120,
        i121,
        i122,
        i123,
        i124,
        i125,
        i126,
        i127,
        i128,
        i129,
        i130,
        i131,
        i132,
        i133,
        i134,
        i135,
        i136,
        i137,
        i138,
        i139,
        i140,
        i141,
        i142,
        i143,
        i144,
        i145,
        i146,
        i147,
        i148,
        i149,
        i150,
        i151,
        i152,
        i153,
        i154,
        i155,
        i156,
        i157,
        i158,
        i159,
        i160,
    ):
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
            (
                i65,
                i66,
            ),
            (
                i67,
                i68,
            ),
            (
                i69,
                i70,
            ),
            (
                i71,
                i72,
            ),
            (
                i73,
                i74,
            ),
            (
                i75,
                i76,
            ),
            (
                i77,
                i78,
            ),
            (
                i79,
                i80,
            ),
            (
                i81,
                i82,
            ),
            (
                i83,
                i84,
            ),
            (
                i85,
                i86,
            ),
            (
                i87,
                i88,
            ),
            (
                i89,
                i90,
            ),
            (
                i91,
                i92,
            ),
            (
                i93,
                i94,
            ),
            (
                i95,
                i96,
            ),
            (
                i97,
                i98,
            ),
            (
                i99,
                i100,
            ),
            (
                i101,
                i102,
            ),
            (
                i103,
                i104,
            ),
            (
                i105,
                i106,
            ),
            (
                i107,
                i108,
            ),
            (
                i109,
                i110,
            ),
            (
                i111,
                i112,
            ),
            (
                i113,
                i114,
            ),
            (
                i115,
                i116,
            ),
            (
                i117,
                i118,
            ),
            (
                i119,
                i120,
            ),
            (
                i121,
                i122,
            ),
            (
                i123,
                i124,
            ),
            (
                i125,
                i126,
            ),
            (
                i127,
                i128,
            ),
            (
                i129,
                i130,
            ),
            (
                i131,
                i132,
            ),
            (
                i133,
                i134,
            ),
            (
                i135,
                i136,
            ),
            (
                i137,
                i138,
            ),
            (
                i139,
                i140,
            ),
            (
                i141,
                i142,
            ),
            (
                i143,
                i144,
            ),
            (
                i145,
                i146,
            ),
            (
                i147,
                i148,
            ),
            (
                i149,
                i150,
            ),
            (
                i151,
                i152,
            ),
            (
                i153,
                i154,
            ),
            (
                i155,
                i156,
            ),
            (
                i157,
                i158,
            ),
            (
                i159,
                i160,
            ),
        )
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


class CombinedModel(torch.nn.Module):
    def __init__(
        self,
        first_vicuna_model_path="TheBloke/vicuna-7B-1.1-HF",
        second_vicuna_model_path="TheBloke/vicuna-7B-1.1-HF",
    ):
        super().__init__()
        self.first_vicuna = FirstVicuna(first_vicuna_model_path)
        # NOT using this path for 13B currently, hence using `SecondVicuna7B`.
        self.second_vicuna = SecondVicuna7B(second_vicuna_model_path)

    def forward(self, input_ids):
        first_output = self.first_vicuna(input_ids=input_ids)
        # generate second vicuna
        compilation_input_ids = torch.zeros([1, 1], dtype=torch.int64)
        pkv = tuple(
            (torch.zeros([1, 32, 19, 128], dtype=torch.float32))
            for _ in range(64)
        )
        secondVicunaCompileInput = (compilation_input_ids,) + pkv
        second_output = self.second_vicuna(*secondVicunaCompileInput)
        return second_output
