from shark.iree_utils._common import (
    check_device_drivers,
    device_driver_info,
    get_supported_device_list,
)
from shark.iree_utils.vulkan_utils import get_vulkan_triple_flag
from shark.sharkdynamo.shark_backend import shark_torchdynamo_backend
from tank.model_utils import get_training_model
from parameterized import parameterized
import torch
import torch.nn as nn
import torch._dynamo as dynamo
import transformers
import iree.compiler as ireec
import pytest
import unittest
import numpy as np
import tempfile
import os
import sys
import copy
import csv


def load_csv_and_convert(filename, gen=False):
    """
    takes in a csv filename and generates a dict for consumption by get_valid_test_params
    """
    model_configs = []
    with open(filename, "r+") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if len(row) < 5:
                print("invalid model: " + row)
                continue
            model_configs.append(
                {
                    "model_name": row[0],
                    "dialect": row[1],
                    "framework": row[2],
                    "rtol": float(row[3]),
                    "atol": float(row[4]),
                    "out_type": row[5],
                    "flags": row[6],
                    "xfail_cpu": row[7],
                    "xfail_cuda": row[8],
                    "xfail_vkm": row[9],
                    "xfail_reason": row[10],
                    "xfail_other": row[11],
                }
            )
    # This is a pytest workaround
    if gen:
        with open(
            os.path.join(os.path.dirname(__file__), "dict_configs.py"), "w+"
        ) as out:
            out.write("ALL = [\n")
            for c in model_configs:
                out.write(str(c) + ",\n")
            out.write("]")
    return model_configs


def get_valid_test_params(custom_device=None):
    """
    Generate a list of all combinations of available devices and static/dynamic flag.
    """
    device_list = [
        device
        for device in get_supported_device_list()
        if not check_device_drivers(device)
    ]
    if custom_device:
        device_list.append(custom_device)
    dynamic_list = (True, False)
    # TODO: This is soooo ugly, but for some reason creating the dict at runtime
    # results in strange pytest failures.
    load_csv_and_convert(
        os.path.join(os.path.dirname(__file__), "all_models.csv"), True
    )
    from tank.dict_configs import ALL

    config_list = ALL

    param_list = [
        (dynamic, device, config)
        for dynamic in dynamic_list
        for device in device_list
        for config in config_list
    ]

    filtered_param_list = [
        params for params in param_list if is_valid_case(params)
    ]

    return filtered_param_list


def is_valid_case(test_params):
    if test_params[0] == True and test_params[2]["framework"] == "tf":
        return False
    elif "fp16" in test_params[2]["model_name"] and test_params[1] != "cuda":
        return False
    else:
        return True


def shark_test_name_func(testcase_func, param_num, param):
    """
    Generate function name string which shows dynamic/static and device name.
    this will be ingested by 'parameterized' package to rename the pytest.
    """
    param_names = []
    for x in param.args:
        if x == True:
            param_names.append("dynamic")
        elif x == False:
            param_names.append("static")
        elif "model" in str(x):
            as_list = str(x).split(" ")
            as_list = [
                parameterized.to_safe_name(x).strip("_") for x in as_list
            ]
            param_names.insert(0, as_list[as_list.index("model_name") + 1])
            param_names.insert(1, as_list[as_list.index("framework") + 1])
            # param_names.append(as_list[3])

        else:
            param_names.append(x)
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param_names)),
    )


class SharkModuleTester:
    def __init__(self, config):
        """config should be a dict containing minimally:
        dialect: (str) name of input dialect
        framework: (str) one of tf, tflite, pytorch
        model_name: (str) name of the model in the tank ("resnet50")
        rtol/atol: (float) tolerances for golden values
        """
        self.config = config

    def create_module_sharkdynamo(self, dynamic, device):
        model_name = self.config["model_name"]
        model_config = {
            "batch_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "hidden_size": 16,
            "vocab_size": 8192,
        }
        net = get_training_model(model_name, model_config)

        in_dim = 128
        out_dim = 8

        input_ids = torch.randint(
            0, 5000, (out_dim, in_dim), dtype=torch.int64
        )
        input_mask = torch.ones([out_dim, in_dim], dtype=torch.int64)
        masked_lm_labels = torch.randint(
            0, 3000, (out_dim, in_dim), dtype=torch.int64
        )
        next_sentence_labels = torch.randint(
            0, 2, (out_dim,), dtype=torch.int64
        )
        segment_ids = torch.randint(0, 2, (out_dim, in_dim), dtype=torch.int64)

        torch.set_grad_enabled(True)
        net.train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)

        def train_func(
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_labels,
            next_sentence_labels,
        ):
            loss = net(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=masked_lm_labels,
                next_sentence_label=next_sentence_labels,
            ).loss
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            return loss

        torch.manual_seed(0)
        print("compiling with dynamo...")
        dynamo_callable = dynamo.optimize(shark_torchdynamo_backend)(
            train_func
        )
        print("running dynamo-compiled module...")
        res = dynamo_callable(
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_labels,
            next_sentence_labels,
        )
        print("res", res)

        # TODO: add baseline for validation
        # baseline_res =


class SharkModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.pytestconfig = pytestconfig
        param_list = get_valid_test_params(
            custom_device=pytestconfig.getoption("custom_device")
        )

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device, config):
        self.module_tester = SharkModuleTester(config)
        self.module_tester.testconfig = self.pytestconfig.args
        safe_name = (
            f"{config['model_name']}_dynamo_pretrain_{dynamic}_{device}"
        )
        self.module_tester.tmp_prefix = safe_name.replace("/", "_")

        tempdir = tempfile.TemporaryDirectory(
            prefix=self.module_tester.tmp_prefix, dir="."
        )
        self.module_tester.temp_dir = tempdir.name

        with ireec.tools.TempFileSaver(tempdir.name):
            self.module_tester.create_module_sharkdynamo(dynamic, device)
