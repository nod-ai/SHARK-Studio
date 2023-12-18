import base64
import io
import os
import time
import datetime
import uvicorn
import ipaddress
import requests
import gradio as gr
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from apps.shark_studio.modules.img_processing import sampler_list
from sdapi_v1 import shark_sd_api
from api.llm import chat_api


def decode_base64_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")

        if opts.api_forbid_local_requests and not verify_url(encoding):
            raise HTTPException(
                status_code=500, detail="Request to local resource not allowed"
            )

        headers = {"user-agent": opts.api_useragent} if opts.api_useragent else {}
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
        if opts.samples_format.lower() == "png":
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
                quality=opts.jpeg_quality,
            )

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            if image.mode == "RGBA":
                image = image.convert("RGB")
            parameters = image.info.get("parameters", None)
            exif_bytes = piexif.dump(
                {
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                            parameters or "", encoding="unicode"
                        )
                    }
                }
            )
            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(
                    output_bytes,
                    format="JPEG",
                    exif=exif_bytes,
                    quality=opts.jpeg_quality,
                )
            else:
                image.save(
                    output_bytes,
                    format="WEBP",
                    exif=exif_bytes,
                    quality=opts.jpeg_quality,
                )

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)


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
        if shared.cmd_opts.api_log and endpoint.startswith("/sdapi"):
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
                errors.report(message, exc_info=True)
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
    def __init__(self, queue_lock: Lock):
        self.router = APIRouter()
        self.app = FastAPI()
        self.queue_lock = queue_lock
        api_middleware(self.app)
        self.add_api_route("/sdapi/v1/txt2img", shark_sd_api, methods=["post"])
        self.add_api_route("/sdapi/v1/img2img", shark_sd_api, methods=["post"])
        # self.add_api_route("/sdapi/v1/upscaler", self.upscaler_api, methods=["post"])
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
        self.add_api_route("/v1/chat/completions", chat_api, methods=["post"])
        self.add_api_route("/v1/completions", chat_api, methods=["post"])
        self.add_api_route("/chat/completions", chat_api, methods=["post"])
        self.add_api_route("/completions", chat_api, methods=["post"])
        self.add_api_route(
            "/v1/engines/codegen/completions", chat_api, methods=["post"]
        )
        if studio.cmd_opts.api_server_stop:
            self.add_api_route(
                "/sdapi/v1/server-kill", self.kill_studio, methods=["POST"]
            )
            self.add_api_route(
                "/sdapi/v1/server-restart",
                self.restart_studio,
                methods=["POST"],
            )
            self.add_api_route(
                "/sdapi/v1/server-stop", self.stop_studio, methods=["POST"]
            )

        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def add_api_route(self, path: str, endpoint, **kwargs):
        if studio.cmd_opts.api_auth:
            return self.app.add_api_route(
                path, endpoint, dependencies=[Depends(self.auth)], **kwargs
            )
        return self.app.add_api_route(path, endpoint, **kwargs)

    def refresh_checkpoints(self):
        with self.queue_lock:
            studio_data.refresh_checkpoints()

    def refresh_vae(self):
        with self.queue_lock:
            studio_data.refresh_vae_list()

    def unloadapi(self):
        unload_model_weights()

        return {}

    def reloadapi(self):
        reload_model_weights()

        return {}

    def skip(self):
        studio.state.skip()

    def launch(self, server_name, port, root_path):
        self.app.include_router(self.router)
        uvicorn.run(
            self.app,
            host=server_name,
            port=port,
            timeout_keep_alive=studio.cmd_opts.timeout_keep_alive,
            root_path=root_path,
        )

    def kill_studio(self):
        restart.stop_program()

    def restart_studio(self):
        if restart.is_restartable():
            restart.restart_program()
        return Response(status_code=501)

    def preprocess(self, args: dict):
        try:
            studio.state.begin(job="preprocess")
            preprocess(**args)
            studio.state.end()
            return models.PreprocessResponse(info="preprocess complete")
        except:
            studio.state.end()

    def stop_studio(request):
        studio.state.server_command = "stop"
        return Response("Stopping.")
