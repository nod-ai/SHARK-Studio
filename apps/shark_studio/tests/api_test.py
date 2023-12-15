# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
from apps.shark_studio.api.llm import LanguageModel


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
