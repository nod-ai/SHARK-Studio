import logging
import traceback
import os
import sys
from apps.stable_diffusion.src import get_available_devices
from apps.stable_diffusion.scripts import txt2img_inf
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram import BotCommand
from telegram.ext import Application, ApplicationBuilder, CallbackQueryHandler
from telegram.ext import ContextTypes, MessageHandler, CommandHandler, filters
from io import BytesIO
import random

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(
        sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_path, relative_path)


log = logging.getLogger("TG.Bot")
logging.basicConfig()
log.warning("Start")
load_dotenv()
if "AMD_ENABLE_LLPC" not in os.environ:
    os.environ["AMD_ENABLE_LLPC"] = "1"

if sys.platform == "darwin":
    os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/lib"
TG_TOKEN = os.getenv("TG_TOKEN")
SELECTED_MODEL = "stabilityai/stable-diffusion-2-1-base"
SELECTED_SCHEDULER = "EulerAncestralDiscrete"
STEPS = 30
HEIGHT = 512
WIDTH = 512
BATCH_SIZE = 1
HF_MODEL_ID = ""
CKPT_LOC = ""
PRECISION = "fp16"
MAX_LENGTH = 64
SAVE_METADATA_TO_JSON = False
SAVE_METADATA_TO_PNG = False
NEGATIVE_PROMPT = (
    "Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra"
    " limbs,Gross proportions,Missing arms,Mutated hands,Long"
    " neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad"
    " anatomy,Cloned face,Malformed limbs,Missing legs,Too many"
    " fingers,blurry, lowres, text, error, cropped, worst quality, low"
    " quality, jpeg artifacts, out of frame, extra fingers, mutated hands,"
    " poorly drawn hands, poorly drawn face, bad anatomy, extra limbs, cloned"
    " face, malformed limbs, missing arms, missing legs, extra arms, extra"
    " legs, fused fingers, too many fingers"
)
GUIDANCE_SCALE = 6
available_devices = get_available_devices()
models_list = [
    "Linaqruf/anything-v3.0",
    "prompthero/openjourney",
    "wavymulder/Analog-Diffusion",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2-1-base",
    "CompVis/stable-diffusion-v1-4",
]
sheds_list = [
    "DDIM",
    "PNDM",
    "LMSDiscrete",
    "DPMSolverMultistep",
    "EulerDiscrete",
    "EulerAncestralDiscrete",
    "SharkEulerDiscrete",
]


def image_to_bytes(image):
    bio = BytesIO()
    bio.name = "image.jpeg"
    image.save(bio, "JPEG")
    bio.seek(0)
    return bio


def get_try_again_markup():
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


def generate_image(prompt):
    seed = random.randint(1, 10000)
    log.warning(SELECTED_MODEL)
    log.warning(STEPS)
    image, text = txt2img_inf(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=seed,
        scheduler=SELECTED_SCHEDULER,
        hf_model_id=HF_MODEL_ID,
        device=available_devices[0],
        height=HEIGHT,
        width=WIDTH,
        batch_size=BATCH_SIZE,
        batch_count=1,
        custom_model=SELECTED_MODEL,
        # ckpt_loc = CKPT_LOC,
        precision=PRECISION,
        max_length=MAX_LENGTH,
        save_metadata_to_json=SAVE_METADATA_TO_JSON,
        save_metadata_to_png=SAVE_METADATA_TO_PNG,
    )

    return image, seed


async def generate_and_send_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    progress_msg = await update.message.reply_text(
        "Generating image...", reply_to_message_id=update.message.message_id
    )
    try:
        im, seed = generate_image(prompt=update.message.text)
        await context.bot.delete_message(
            chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
        )
    except Exception:
        log.exception("Exception")
        await update.message.reply_text(traceback.format_exc()[:4096])
        return
    await context.bot.send_photo(
        update.effective_user.id,
        image_to_bytes(im[0]),
        caption=f'"{update.message.text}" (Seed: {seed})',
        reply_markup=get_try_again_markup(),
        reply_to_message_id=update.message.message_id,
    )


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query.data in models_list:
        global SELECTED_MODEL
        SELECTED_MODEL = query.data
        await query.answer()
        await query.edit_message_text(text=f"Selected model: {query.data}")
        return
    if query.data in sheds_list:
        global SELECTED_SCHEDULER
        SELECTED_SCHEDULER = query.data
        await query.answer()
        await query.edit_message_text(text=f"Selected scheduler: {query.data}")
        return
    replied_message = query.message.reply_to_message
    await query.answer()
    progress_msg = await query.message.reply_text(
        "Generating image...", reply_to_message_id=replied_message.message_id
    )

    if query.data == "TRYAGAIN":
        prompt = replied_message.text
        try:
            im, seed = generate_image(prompt)
        except Exception:
            log.exception("Exception")
            await query.message.reply_text(traceback.format_exc()[:4096])
            return
    await context.bot.delete_message(
        chat_id=progress_msg.chat_id, message_id=progress_msg.message_id
    )
    await context.bot.send_photo(
        update.effective_user.id,
        image_to_bytes(im[0]),
        caption=f'"{prompt}" (Seed: {seed})',
        reply_markup=get_try_again_markup(),
        reply_to_message_id=replied_message.message_id,
    )


async def select_model_handler(update, context):
    text = "Select model"
    keyboard = []
    for model in models_list:
        keyboard.append(
            [
                InlineKeyboardButton(text=model, callback_data=model),
            ]
        )
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text=text, reply_markup=markup)


async def select_scheduler_handler(update, context):
    text = "Select schedule"
    keyboard = []
    for shed in sheds_list:
        keyboard.append(
            [
                InlineKeyboardButton(text=shed, callback_data=shed),
            ]
        )
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text=text, reply_markup=markup)


async def set_steps_handler(update, context):
    input_mex = update.message.text
    log.warning(input_mex)
    try:
        input_args = input_mex.split("/set_steps ")[1]
        global STEPS
        STEPS = int(input_args)
    except Exception:
        input_args = (
            "Invalid parameter for command. Correct command looks like\n"
            " /set_steps 30"
        )
    await update.message.reply_text(input_args)


async def set_negative_prompt_handler(update, context):
    input_mex = update.message.text
    log.warning(input_mex)
    try:
        input_args = input_mex.split("/set_negative_prompt ")[1]
        global NEGATIVE_PROMPT
        NEGATIVE_PROMPT = input_args
    except Exception:
        input_args = (
            "Invalid parameter for command. Correct command looks like\n"
            " /set_negative_prompt ugly, bad art, mutated"
        )
    await update.message.reply_text(input_args)


async def set_guidance_scale_handler(update, context):
    input_mex = update.message.text
    log.warning(input_mex)
    try:
        input_args = input_mex.split("/set_guidance_scale ")[1]
        global GUIDANCE_SCALE
        GUIDANCE_SCALE = int(input_args)
    except Exception:
        input_args = (
            "Invalid parameter for command. Correct command looks like\n"
            " /set_guidance_scale 7"
        )
    await update.message.reply_text(input_args)


async def setup_bot_commands(application: Application) -> None:
    await application.bot.set_my_commands(
        [
            BotCommand("select_model", "to select model"),
            BotCommand("select_scheduler", "to select scheduler"),
            BotCommand("set_steps", "to set steps"),
            BotCommand("set_guidance_scale", "to set guidance scale"),
            BotCommand("set_negative_prompt", "to set negative prompt"),
        ]
    )


app = (
    ApplicationBuilder().token(TG_TOKEN).post_init(setup_bot_commands).build()
)
app.add_handler(CommandHandler("select_model", select_model_handler))
app.add_handler(CommandHandler("select_scheduler", select_scheduler_handler))
app.add_handler(CommandHandler("set_steps", set_steps_handler))
app.add_handler(
    CommandHandler("set_guidance_scale", set_guidance_scale_handler)
)
app.add_handler(
    CommandHandler("set_negative_prompt", set_negative_prompt_handler)
)
app.add_handler(
    MessageHandler(filters.TEXT & ~filters.COMMAND, generate_and_send_photo)
)
app.add_handler(CallbackQueryHandler(button))
log.warning("Start bot")
app.run_polling()
