# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import json

from apps.shark_studio.api.llm import LanguageModel
from apps.shark_studio.api.sd import shark_sd_fn_dict_input, view_json_file
from apps.shark_studio.web.utils.file_utils import get_resource_path

class SDAPITest(unittest.TestCase):
    def testSDSimple(self):
        from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
        import apps.shark_studio.web.utils.globals as global_obj

        global_obj._init()

        sd_json = view_json_file(get_resource_path("../configs/default_sd_config.json"))
        sd_kwargs = json.loads(sd_json)
        for arg in vars(cmd_opts):
            if arg in sd_kwargs:
                sd_kwargs[arg] = getattr(cmd_opts, arg)
        for i in shark_sd_fn_dict_input(sd_kwargs):
            print(i)

class LLMAPITest(unittest.TestCase):
    def testLLMSimple(self):
        lm = LanguageModel(
            "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            hf_auth_token=None,
            device="cpu-task",
            external_weights="safetensors",
        )
        count = 0
        for msg, _ in lm.chat("hi, what are you?"):
            # skip first token output
            if count == 0:
                count += 1
                continue
            assert (
                msg.strip(" ") == "Hello"
            ), f"LLM API failed to return correct response, expected 'Hello', received {msg}"
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
