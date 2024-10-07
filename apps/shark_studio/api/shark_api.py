# from turbine_models.custom_models import stateless_llama
from turbine_models.model_runner import vmfbRunner

# from turbine_models.gen_external_params.gen_external_params import gen_external_params
from shark.iree_utils.compile_utils import compile_module_to_flatbuffer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from sharktank.layers import *
from sharktank.types import *
from shark_turbine.aot import *

# from sharktank.models.mixtral.mixtral import *
from sharktank.models.llama.llama import *
from sharktank.utils.debugging import trace_tensor
from sharktank.utils.tokenizer import InferenceTokenizer, load_tokenizer

from shark_turbine.aot import *

# from sharktank.models.llama.llama import LlamaModelConfig, PagedLlamaModelV1
import sharktank

# Internal API
sd_pipelines = {
    "sd1.5": ("", None),
    "sd2": ("", None),
    "sdxl": ("", None),
    "sd3": ("", None),
}
language_models = {}
system_prompt = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n <</SYS>>\n\n
"""


# Used for filenames as well as the key for the global cache
def safe_name(
    model_name: str,
    height: int,
    width: int,
    batch_size: int,
):
    pass


def local_path():
    pass


# External API
def generate_images(
    prompt: str,
    negative_prompt: str,
    *,
    height: int = 512,
    width: int = 512,
    steps: int = 20,
    strength: float = 0.8,
    sd_init_image: list = None,
    guidance_scale: float = 7.5,
    seed: list = -1,
    batch_count: int = 1,
    batch_size: int = 1,
    scheduler: str = "EulerDiscrete",
    base_model: str = "sd2",
    custom_weights: str = None,
    custom_vae: str = None,
    precision: str = "fp16",
    device: str = "cpu",
    target_triple: str = None,
    ondemand: bool = False,
    compiled_pipeline: bool = False,
    resample_type: str = "Nearest Neighbor",
    controlnets: dict = {},
    embeddings: dict = {},
    **kwargs,
):
    sd_kwargs = locals()

    # Handle img2img
    if not isinstance(sd_init_image, list):
        sd_init_image = [sd_init_image] * batch_count
    is_img2img = True if sd_init_image[0] is not None else False

    # Generate seed if < 0
    # TODO

    # Cache dir
    # TODO
    pipeline_dir = None

    # Sanity checks
    assert scheduler in ["EulerDiscrete"]
    assert base_model in ["sd1.5", "sd2", "sdxl", "sd3"]
    assert precision in ["fp16", "fp32"]
    assert device in [
        "cpu",
        "vulkan",
        "rocm",
        "hip",
        "cuda",
    ]  # and (IREE check if the device exists)
    assert resample_type in ["Nearest Neighbor"]

    # Custom weights
    # TODO
    # Custom VAE
    # TODO
    # Target triple
    # TODO

    # (Re)initialize pipeline
    pipeline_args = {
        "height": height,
        "width": width,
        "batch_size": batch_size,
        "precision": precision,
        "device": device,
        "target_triple": target_triple,
    }
    (existing_args, pipeline) = sd_pipelines[base_model]
    if not existing_args or not pipeline or not pipeline_args == existing_args:
        # TODO: Initialize new pipeline
        if base_model in ["sd1.5", "sd2"]:
            new_pipeline = SharkSDPipeline(
                hf_model_name=(
                    "stabilityai/stable-diffusion-2-1"
                    if base_model == "sd2"
                    else "stabilityai/stable-diffusion-1-5"
                ),
                scheduler_id=scheduler,
                height=height,
                width=width,
                precision=precision,
                max_length=64,
                batch_size=batch_size,
                num_inference_steps=steps,
                device=device,  # TODO: Get the IREE device ID?
                iree_target_triple=target_triple,
                ireec_flags={},
                attn_spec=None,  # TODO: Find a better way to figure this out than hardcoding
                decomp_attn=True,  # TODO: Ditto
                pipeline_dir=pipeline_dir,
                external_weights_dir=weights,  # TODO: Are both necessary still?
                external_weights=weights,
                custom_vae=custom_vae,
            )
        elif base_model == "sdxl":
            pass
        elif base_model == "sd3":
            pass
        existing_args = pipeline_args
        pipeline = new_pipeline
        sd_pipelines[base_model] = (existing_args, pipeline)

    generated_images = []
    for current_batch in range(batch_count):

        start_time = time.time()
        for t in range(steps):

            out_images = pipeline.generate_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=sd_init_image[current_batch],
                strength=strength,
                guidance_scale=guidance_scale,
                seed=seed,
                ondemand=ondemand,
                resample_type=resample_type,
                control_mode=control_mode,
                hints=hints,
            )

        # Processing time
        total_time = time.time() - start_time
        # text_output = f"Total image(s) generation time: {total_time:.4f}sec"
        # print(f"\n[LOG] {text_output}")

        # TODO: Add to output list
        if not isinstance(out_images, list):
            out_images = [out_images]
        generated_images.extend(out_images)

        # TODO: Allow the user to halt the process

    return generated_images


def chat(
    prompt,
    model_name,
    history: list = [],
    hf_auth_token: str = None,
    device=None,
    target_triple=None,
    max_tokens=4096,
    quantization="int4",
    precision="f16",
    external_weights=None,
    use_system_prompt=True,
    streaming_llm=False,
    batch_sizes=[4],
):
    # Compile model if necessary
    if not model_name in language_models or language_models[model_name] is None:
        language_models[model_name] = None
        # gen_external_params(
        #     hf_model_name=model_name,
        #     quantization=quantization,
        #     weight_path="llama.safetensors",
        #     hf_auth_token=hf_auth_token,
        #     precision=precision,
        # )
        # torch_ir, _ = stateless_llama.export_transformer_model(
        #     model_name,
        #     hf_auth_token,
        #     compile_to="torch",
        #     external_weights=None, #external_weights="llama.safetensors",
        #     precision=precision,
        #     quantization=quantization,
        #     streaming_llm=streaming_llm,
        #     decomp_attn=True,
        # )

        import pdb

        pdb.set_trace()
        dataset = sharktank.types.Dataset.load("llama.gguf", file_type="gguf")
        hp = sharktank.layers.configs.LlamaHParams.from_gguf_props(dataset.properties)
        llama_config = sharktank.models.llama.llama.LlamaModelConfig(hp)
        llama_config.kv_cache_type = "paged"
        model = PagedLlamaModelV1(dataset.root_theta, llama_config)

        def generate_params_json(hp, prefill_bs: list[int], decode_bs: list[int]):
            return {
                "module_name": "module",
                "module_abi_version": 1,
                "max_seq_len": hp.context_length,
                "attn_head_count": hp.attention_head_count,
                "attn_head_dim": hp.attn_head_dim,
                "prefill_batch_sizes": prefill_bs,
                "decode_batch_sizes": decode_bs,
                "transformer_block_count": hp.block_count,
                "block_seq_stride": llama_config.block_seq_stride,
            }

        import torch._dynamo.config as dynamo_config

        fxb = FxProgramsBuilder(model)

        def generate_batch_prefill(bs: int):
            tokens = torch.empty(bs, 64, dtype=torch.int64)
            seq_lens = torch.empty(bs, dtype=torch.int64)
            seq_block_ids = torch.empty(bs, 4, dtype=torch.int64)
            block_dim = torch.export.Dim(
                "block", max=(hp.context_length - 1) // llama_config.block_seq_stride
            )
            import pdb

            pdb.set_trace()
            sl_dim = llama_config.block_seq_stride * block_dim

            if model.config.kv_cache_type == "paged":
                cache_state = model.cache.allocate(
                    page_count=hp.context_length // llama_config.block_seq_stride
                )
                page_dim = torch.export.Dim("page")
                cache_state_dynamic_shapes = [{0: page_dim}]
            elif model.config.kv_cache_type == "direct":
                cache_state = model.cache.allocate(bs=1)
                # Direct cache dimensions:
                #   2 * transformer_block_count of...
                #   [bs, seq_length, attn_head_count, attn_head_dim]
                cache_state_dynamic_shapes = (2 * hp.block_count) * [{}]
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache type: {type(model.cache)}"
                )

            dynamic_shapes = {
                "tokens": {1: sl_dim},
                "seq_lens": {},
                "seq_block_ids": {1: block_dim},
                "cache_state": cache_state_dynamic_shapes,
            }

            print(f"Exporting prefill_bs{bs}")

            @fxb.export_program(
                name=f"prefill_bs{bs}",
                args=(tokens, seq_lens, seq_block_ids, cache_state),
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )
            def _(model, tokens, seq_lens, seq_block_ids, cache_state):
                sl = tokens.shape[1]
                input_mask = model.input_mask(seq_lens, sl)
                attention_mask = model.attention_mask(input_mask)
                logits = model.prefill(
                    tokens,
                    attention_mask=attention_mask,
                    seq_block_ids=seq_block_ids,
                    cache_state=cache_state,
                )
                return logits

        def generate_batch_decode(bs: int):
            tokens = torch.ones(bs, 1, dtype=torch.int64)
            seq_lens = torch.ones(bs, dtype=torch.int64)
            start_positions = torch.ones(bs, dtype=torch.int64)
            seq_block_ids = torch.zeros(bs, 4, dtype=torch.int64)
            block_dim = torch.export.Dim(
                "block", max=(hp.context_length - 1) // llama_config.block_seq_stride
            )

            if model.config.kv_cache_type == "paged":
                cache_state = model.cache.allocate(
                    page_count=hp.context_length // llama_config.block_seq_stride
                )
                page_dim = torch.export.Dim("page")
                cache_state_dynamic_shapes = [{0: page_dim}]
            elif model.config.kv_cache_type == "direct":
                cache_state = model.cache.allocate(bs=1)
                # Direct cache dimensions:
                #   2 * transformer_block_count of...
                #   [bs, seq_length, attn_head_count, attn_head_dim]
                cache_state_dynamic_shapes = (2 * hp.block_count) * [{}]
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache type: {type(model.cache)}"
                )

            dynamic_shapes = {
                "tokens": {},
                "seq_lens": {},
                "start_positions": {},
                "seq_block_ids": {1: block_dim},
                "cache_state": cache_state_dynamic_shapes,
            }

            print(f"Exporting decode_bs{bs}")

            @fxb.export_program(
                name=f"decode_bs{bs}",
                args=(
                    tokens,
                    seq_lens,
                    start_positions,
                    seq_block_ids,
                    cache_state,
                ),
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )
            def _(
                model,
                tokens,
                seq_lens,
                start_positions,
                seq_block_ids,
                cache_state,
            ):
                input_mask = model.input_mask(
                    seq_lens, seq_block_ids.shape[1] * model.cache.block_seq_stride
                )
                attention_mask = model.decode_attention_mask(input_mask)
                logits = model.decode(
                    tokens,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    seq_block_ids=seq_block_ids,
                    cache_state=cache_state,
                )
                return logits

        bsizes = []
        for batch_size in batch_sizes:
            generate_batch_prefill(bs)
            generate_batch_decode(bs)
            bsizes.append(batch_size)
        config = generate_params_json(hp, bsizes, bsizes)

        torch_ir = export(fxb)
        torch_ir.save_mlir("llama.mlir")

        # with open("llama.mlir", "w+") as f:
        #     f.write(torch_ir)
        del torch_ir
        flags = []
        if "cpu" in device:
            flags.extend(
                [
                    "--iree-global-opt-enable-quantized-matmul-reassociation",
                ]
            )
        elif device == "vulkan":
            flags.extend(["--iree-stream-resource-max-allocation-size=4294967296"])
        elif device == "rocm":
            flags.extend(
                [
                    "--iree-codegen-llvmgpu-enable-transform-dialect-jit=false",
                    "--iree-llvmgpu-enable-prefetch=true",
                    "--iree-opt-outer-dim-concat=true",
                    "--iree-flow-enable-aggressive-fusion",
                ]
            )
            # if "gfx9" in target_triple:
            #     flags.extend(
            #         [
            #             f"--iree-codegen-transform-dialect-library={get_mfma_spec_path(target_triple, get_checkpoints_path())}",
            #             "--iree-codegen-llvmgpu-use-vector-distribution=true",
            #         ]
            #     )
        flags.extend(
            [
                "--iree-opt-const-expr-hoisting=False",
                f"--iree-rocm-target-chip={target_triple}",
            ]
        )
        flatbuffer_blob = compile_module_to_flatbuffer(
            "llama.mlir",
            device=device,
            frontend="auto",
            model_config_path=None,
            extra_args=flags,
            write_to="llama.vmfb",
        )
    model = language_models[model_name]
    runner = vmfbRunner(
        device=device,
        vmfb_path="llama.vmfb",  # safe_name
        external_weight_path="llama.safetensors",  # self.external_weight_file,
    )

    # Sanitize prompt
    if isinstance(prompt, list):
        prompt = list(chain.from_iterable(prompt))
        prompt = " ".join([x for x in prompt if isinstance(x, str)])
    prompt = prompt.replace("\n", " ")
    prompt = prompt.replace("\t", " ")
    prompt = prompt.replace("\r", " ")
    if use_system_prompt and not history:
        prompt = append_user_prompt(DEFAULT_CHAT_SYS_PROMPT, prompt)
    else:
        prompt = f"[INST] {prompt} [/INST]"

    # Parse input
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        use_auth_token=hf_auth_token,
    )
    input_tensor = tokenizer(prompt, return_tensors="pt").input_ids

    def format_out(results):
        return torch.tensor(results.to_host()[0][0])

    for iter in range(max_tokens):
        if streaming_llm:
            # token_slice = max(self.prev_token_len - 1, 0)
            token_slice = max(len(history) - 1, 0)
            input_tensor = input_tensor[:, token_slice:]
        if streaming_llm and model["get_seq_step"]() > 600:
            print("Evicting cache space!")
            model["evict_kvcache_space"]()
        token_len = input_tensor.shape[-1]
        device_inputs = [ireert.asdevicearray(runner.config.device, input_tensor)]
        # if self.first_input or not streaming_llm:
        if not history or not streaming_llm:
            st_time = time.time()
            token = model["run_initialize"](*device_inputs)
            total_time = time.time() - st_time
            token_len += 1
            # self.first_input = False
        else:
            st_time = time.time()
            token = model["run_cached_initialize"](*device_inputs)
            total_time = time.time() - st_time
            token_len += 1

        history.append(format_out(token))
        while (
            format_out(token) != llm_model_map[model_name]["stop_token"]
            and len(history) < max_tokens
        ):
            dec_time = time.time()
            if streaming_llm and model["get_seq_step"]() > 600:
                print("Evicting cache space!")
                model["evict_kvcache_space"]()
            token = model["run_forward"](token)
            history.append(format_out(token))
            total_time = time.time() - dec_time
            yield tokenizer.decode(history), total_time

        # self.prev_token_len = token_len + len(history)
        history.append(token)

        if format_out(token) == llm_model_map[model_name]["stop_token"]:
            break

    for i in range(len(history)):
        if type(history[i]) != int:
            history[i] = int(history[i])
    result_output = tokenizer.decode(history)
    # self.global_iter += 1
    return result_output, history, total_time


if __name__ == "__main__":
    output, history, time = chat(
        "Hello.",
        "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        history=[],
        hf_auth_token=None,
        device="rocm",
        target_triple="gfx942",
        max_tokens=4096,
        quantization="int4",
        precision="f16",
        external_weights=None,
        use_system_prompt=True,
        streaming_llm=True,
    )
