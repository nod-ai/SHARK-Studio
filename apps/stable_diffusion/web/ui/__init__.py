from apps.stable_diffusion.web.ui.txt2img_ui import (
    txt2img_inf,
    txt2img_web,
    txt2img_custom_model,
    txt2img_gallery,
    txt2img_png_info_img,
    txt2img_status,
    txt2img_sendto_img2img,
    txt2img_sendto_inpaint,
    txt2img_sendto_outpaint,
    txt2img_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.txt2img_sdxl_ui import (
    txt2img_sdxl_inf,
    txt2img_sdxl_web,
    txt2img_sdxl_custom_model,
    txt2img_sdxl_gallery,
    txt2img_sdxl_status,
)
from apps.stable_diffusion.web.ui.img2img_ui import (
    img2img_inf,
    img2img_web,
    img2img_custom_model,
    img2img_gallery,
    img2img_init_image,
    img2img_status,
    img2img_sendto_inpaint,
    img2img_sendto_outpaint,
    img2img_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.inpaint_ui import (
    inpaint_inf,
    inpaint_web,
    inpaint_custom_model,
    inpaint_gallery,
    inpaint_init_image,
    inpaint_status,
    inpaint_sendto_img2img,
    inpaint_sendto_outpaint,
    inpaint_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.outpaint_ui import (
    outpaint_inf,
    outpaint_web,
    outpaint_custom_model,
    outpaint_gallery,
    outpaint_init_image,
    outpaint_status,
    outpaint_sendto_img2img,
    outpaint_sendto_inpaint,
    outpaint_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.upscaler_ui import (
    upscaler_inf,
    upscaler_web,
    upscaler_custom_model,
    upscaler_gallery,
    upscaler_init_image,
    upscaler_status,
    upscaler_sendto_img2img,
    upscaler_sendto_inpaint,
    upscaler_sendto_outpaint,
)
from apps.stable_diffusion.web.ui.model_manager import (
    model_web,
    hf_models,
    modelmanager_sendto_txt2img,
    modelmanager_sendto_img2img,
    modelmanager_sendto_inpaint,
    modelmanager_sendto_outpaint,
    modelmanager_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.lora_train_ui import lora_train_web
from apps.stable_diffusion.web.ui.stablelm_ui import (
    stablelm_chat,
    llm_chat_api,
)
from apps.stable_diffusion.web.ui.generate_config import model_config_web
from apps.stable_diffusion.web.ui.minigpt4_ui import minigpt4_web
from apps.stable_diffusion.web.ui.outputgallery_ui import (
    outputgallery_web,
    outputgallery_tab_select,
    outputgallery_watch,
    outputgallery_filename,
    outputgallery_sendto_txt2img,
    outputgallery_sendto_img2img,
    outputgallery_sendto_inpaint,
    outputgallery_sendto_outpaint,
    outputgallery_sendto_upscaler,
)
