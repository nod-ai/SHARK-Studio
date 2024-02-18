import base64
import io
import os
import time
import datetime
import uvicorn
import ipaddress
import requests
import threading
import collections
import gradio as gr
from PIL import Image, PngImagePlugin
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from apps.shark_studio.modules.shared_cmd_opts import cmd_opts

# from sdapi_v1 import shark_sd_api
from apps.shark_studio.api.llm import llm_chat_api


def decode_base64_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):

        headers = {}
        response = requests.get(encoding, timeout=30, headers=headers)
        try:
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid image url") from e

    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:
        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True
        image.save(
            output_bytes,
            format="PNG",
            pnginfo=(metadata if use_metadata else None),
        )

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


# reference: https://gist.github.com/vitaliyp/6d54dd76ca2c3cdfc1149d33007dc34a
class FIFOLock(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._inner_lock = threading.Lock()
        self._pending_threads = collections.deque()

    def acquire(self, blocking=True):
        with self._inner_lock:
            lock_acquired = self._lock.acquire(False)
            if lock_acquired:
                return True
            elif not blocking:
                return False

            release_event = threading.Event()
            self._pending_threads.append(release_event)

        release_event.wait()
        return self._lock.acquire()

    def release(self):
        with self._inner_lock:
            if self._pending_threads:
                release_event = self._pending_threads.popleft()
                release_event.set()

            self._lock.release()

    __enter__ = acquire

    def __exit__(self, t, v, tb):
        self.release()


def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get("WEBUI_RICH_EXCEPTIONS", None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console

            console = Console()
            rich_available = True
    except Exception:
        pass

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get("path", "err")
        if cmd_opts.api_log and endpoint.startswith("/sdapi"):
            print(
                "API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}".format(
                    t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    code=res.status_code,
                    ver=req.scope.get("http_version", "0.0"),
                    cli=req.scope.get("client", ("0:0.0.0", 0))[0],
                    prot=req.scope.get("scheme", "err"),
                    method=req.scope.get("method", "err"),
                    endpoint=endpoint,
                    duration=duration,
                )
            )
        return res

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get("detail", ""),
            "body": vars(e).get("body", ""),
            "errors": str(e),
        }
        if not isinstance(
            e, HTTPException
        ):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(
                    show_locals=True,
                    max_frames=2,
                    extra_lines=1,
                    suppress=[anyio, starlette],
                    word_wrap=False,
                    width=min([console.width, 200]),
                )
            else:
                print(message)
                raise (e)
        return JSONResponse(
            status_code=vars(e).get("status_code", 500),
            content=jsonable_encoder(err),
        )

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class ApiCompat:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        api_middleware(self.app)
        # self.add_api_route("/sdapi/v1/txt2img", shark_sd_api, methods=["POST"])
        # self.add_api_route("/sdapi/v1/img2img", shark_sd_api, methods=["POST"])
        # self.add_api_route("/sdapi/v1/upscaler", self.upscaler_api, methods=["POST"])
        # self.add_api_route("/sdapi/v1/extra-single-image", self.extras_single_image_api, methods=["POST"], response_model=models.ExtrasSingleImageResponse)
        # self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_api, methods=["POST"], response_model=models.ExtrasBatchImagesResponse)
        # self.add_api_route("/sdapi/v1/png-info", self.pnginfoapi, methods=["POST"], response_model=models.PNGInfoResponse)
        # self.add_api_route("/sdapi/v1/progress", self.progressapi, methods=["GET"], response_model=models.ProgressResponse)
        # self.add_api_route("/sdapi/v1/interrogate", self.interrogateapi, methods=["POST"])
        # self.add_api_route("/sdapi/v1/interrupt", self.interruptapi, methods=["POST"])
        # self.add_api_route("/sdapi/v1/skip", self.skip, methods=["POST"])
        # self.add_api_route("/sdapi/v1/options", self.get_config, methods=["GET"], response_model=models.OptionsModel)
        # self.add_api_route("/sdapi/v1/options", self.set_config, methods=["POST"])
        # self.add_api_route("/sdapi/v1/cmd-flags", self.get_cmd_flags, methods=["GET"], response_model=models.FlagsModel)
        # self.add_api_route("/sdapi/v1/samplers", self.get_samplers, methods=["GET"], response_model=List[models.SamplerItem])
        # self.add_api_route("/sdapi/v1/upscalers", self.get_upscalers, methods=["GET"], response_model=List[models.UpscalerItem])
        # self.add_api_route("/sdapi/v1/latent-upscale-modes", self.get_latent_upscale_modes, methods=["GET"], response_model=List[models.LatentUpscalerModeItem])
        # self.add_api_route("/sdapi/v1/sd-models", self.get_sd_models, methods=["GET"], response_model=List[models.SDModelItem])
        # self.add_api_route("/sdapi/v1/sd-vae", self.get_sd_vaes, methods=["GET"], response_model=List[models.SDVaeItem])
        # self.add_api_route("/sdapi/v1/hypernetworks", self.get_hypernetworks, methods=["GET"], response_model=List[models.HypernetworkItem])
        # self.add_api_route("/sdapi/v1/face-restorers", self.get_face_restorers, methods=["GET"], response_model=List[models.FaceRestorerItem])
        # self.add_api_route("/sdapi/v1/realesrgan-models", self.get_realesrgan_models, methods=["GET"], response_model=List[models.RealesrganItem])
        # self.add_api_route("/sdapi/v1/prompt-styles", self.get_prompt_styles, methods=["GET"], response_model=List[models.PromptStyleItem])
        # self.add_api_route("/sdapi/v1/embeddings", self.get_embeddings, methods=["GET"], response_model=models.EmbeddingsResponse)
        # self.add_api_route("/sdapi/v1/refresh-checkpoints", self.refresh_checkpoints, methods=["POST"])
        # self.add_api_route("/sdapi/v1/refresh-vae", self.refresh_vae, methods=["POST"])
        # self.add_api_route("/sdapi/v1/create/embedding", self.create_embedding, methods=["POST"], response_model=models.CreateResponse)
        # self.add_api_route("/sdapi/v1/create/hypernetwork", self.create_hypernetwork, methods=["POST"], response_model=models.CreateResponse)
        # self.add_api_route("/sdapi/v1/preprocess", self.preprocess, methods=["POST"], response_model=models.PreprocessResponse)
        # self.add_api_route("/sdapi/v1/train/embedding", self.train_embedding, methods=["POST"], response_model=models.TrainResponse)
        # self.add_api_route("/sdapi/v1/train/hypernetwork", self.train_hypernetwork, methods=["POST"], response_model=models.TrainResponse)
        # self.add_api_route("/sdapi/v1/memory", self.get_memory, methods=["GET"], response_model=models.MemoryResponse)
        # self.add_api_route("/sdapi/v1/unload-checkpoint", self.unloadapi, methods=["POST"])
        # self.add_api_route("/sdapi/v1/reload-checkpoint", self.reloadapi, methods=["POST"])
        # self.add_api_route("/sdapi/v1/scripts", self.get_scripts_list, methods=["GET"], response_model=models.ScriptsList)
        # self.add_api_route("/sdapi/v1/script-info", self.get_script_info, methods=["GET"], response_model=List[models.ScriptInfo])

        # chat APIs needed for compatibility with multiple extensions using OpenAI API
        self.add_api_route("/v1/chat/completions", llm_chat_api, methods=["POST"])
        self.add_api_route("/v1/completions", llm_chat_api, methods=["POST"])
        self.add_api_route("/chat/completions", llm_chat_api, methods=["POST"])
        self.add_api_route("/completions", llm_chat_api, methods=["POST"])
        self.add_api_route(
            "/v1/engines/codegen/completions", llm_chat_api, methods=["POST"]
        )

        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    # def refresh_checkpoints(self):
    #     with self.queue_lock:
    #         studio_data.refresh_checkpoints()

    # def refresh_vae(self):
    #     with self.queue_lock:
    #         studio_data.refresh_vae_list()

    # def unloadapi(self):
    #     unload_model_weights()

    #     return {}

    # def reloadapi(self):
    #     reload_model_weights()

    #     return {}

    # def skip(self):
    #     studio.state.skip()

    def launch(self, server_name, port, root_path):
        self.app.include_router(self.router)
        uvicorn.run(
            self.app,
            host=server_name,
            port=port,
            root_path=root_path,
        )

    # def kill_studio(self):
    #     restart.stop_program()

    # def restart_studio(self):
    #     if restart.is_restartable():
    #         restart.restart_program()
    #     return Response(status_code=501)

    # def preprocess(self, args: dict):
    #     try:
    #         studio.state.begin(job="preprocess")
    #         preprocess(**args)
    #         studio.state.end()
    #         return models.PreprocessResponse(info="preprocess complete")
    #     except:
    #         studio.state.end()

    # def stop_studio(request):
    #     studio.state.server_command = "stop"
    #     return Response("Stopping.")
