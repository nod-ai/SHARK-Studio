# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import torch
import unittest
import PIL
from typing import List
from apps.shark_studio.api.sd import SharkDiffusionPipeline
#from diffusers import DiffusionPipeline


class SDBaseAPITest(unittest.TestCase):
    def testPipeSimple(self):
        pipe = SharkDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1",
            device="vulkan",
            torch_dtype=torch.float16,
        ).to(torch.float16)
        pipe.setup_shark(
            base_model_id="stabilityai/stable-diffusion-2-1",
            height=512,
            width=512,
            batch_size=1,
            precision="f16",
            device="vulkan",
        )
        pipe.prepare_pipe(
            custom_weights="",
            adapters=[],
            embeddings=[],
            is_img2img=False,
        )

        prompt = ["An astronaut riding a fearsome shark"]
        negative_prompt = [""]
        image = pipe(prompt, negative_prompt).images[0]
        assert isinstance(image, List(PIL.Image.Image))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()