from apps.stable_diffusion.web.ui.txt2img_ui import (
    txt2img_api,
    txt2img_web,
    txt2img_gallery,
    txt2img_sendto_img2img,
    txt2img_sendto_inpaint,
    txt2img_sendto_outpaint,
    txt2img_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.img2img_ui import (
    img2img_api,
    img2img_web,
    img2img_gallery,
    img2img_init_image,
    img2img_sendto_inpaint,
    img2img_sendto_outpaint,
    img2img_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.inpaint_ui import (
    inpaint_api,
    inpaint_web,
    inpaint_gallery,
    inpaint_init_image,
    inpaint_sendto_img2img,
    inpaint_sendto_outpaint,
    inpaint_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.outpaint_ui import (
    outpaint_api,
    outpaint_web,
    outpaint_gallery,
    outpaint_init_image,
    outpaint_sendto_img2img,
    outpaint_sendto_inpaint,
    outpaint_sendto_upscaler,
)
from apps.stable_diffusion.web.ui.upscaler_ui import (
    upscaler_api,
    upscaler_web,
    upscaler_gallery,
    upscaler_init_image,
    upscaler_sendto_img2img,
    upscaler_sendto_inpaint,
    upscaler_sendto_outpaint,
)
from apps.stable_diffusion.web.ui.lora_train_ui import lora_train_web
