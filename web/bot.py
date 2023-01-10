import logging
import torch
from torch import autocast
from PIL import Image
import os
from models.stable_diffusion.resources import resource_path, prompt_examples
from models.stable_diffusion.main import stable_diff_inf
from models.stable_diffusion.stable_args import args
from models.stable_diffusion.utils import get_available_devices
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, CommandHandler, filters
from io import BytesIO
import random
log = logging.getLogger("TG.Bot")
logging.basicConfig()
log.warning('Start')
load_dotenv()
os.environ["AMD_ENABLE_LLPC"] = "0"
TG_TOKEN = os.getenv('TG_TOKEN')
SELECTED_MODEL = 'stablediffusion'
SELECTED_SCHEDULER = 'EulerAncestralDiscrete'
STEPS = 30
NEGATIVE_PROMPT = 'Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers,blurry, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, bad anatomy, extra limbs, cloned face, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers'
GUIDANCE_SCALE = 6
available_devices = get_available_devices()
models_list = ["stablediffusion","anythingv3","analogdiffusion","openjourney","dreamlike"]
sheds_list = ["DDIM","PNDM","LMSDiscrete","DPMSolverMultistep","EulerDiscrete","EulerAncestralDiscrete","SharkEulerDiscrete",]

def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
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
    image, text = stable_diff_inf(
    prompt=prompt,
    negative_prompt=NEGATIVE_PROMPT,
    steps=STEPS,
    guidance_scale=GUIDANCE_SCALE,
    seed=seed,
    scheduler_key=SELECTED_SCHEDULER,
    variant=SELECTED_MODEL,
    device_key=available_devices[0],
	)

    return image, seed


async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = generate_image(prompt=update.message.text)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_user.id, image_to_bytes(im), caption=f'"{update.message.text}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)


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
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)

    if query.data == "TRYAGAIN":
            prompt = replied_message.text
            im, seed = generate_image(prompt)

    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_user.id, image_to_bytes(im), caption=f'"{prompt}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)

async def select_model_handler(update, context):
    text = 'Select model'
    keyboard = []
    for model in models_list:
        keyboard.append([InlineKeyboardButton(text=model, callback_data=model),])
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text=text, reply_markup=markup)
async def select_scheduler_handler(update, context):
    text = 'Select schedule'
    keyboard = []
    for shed in sheds_list:
        keyboard.append([InlineKeyboardButton(text=shed, callback_data=shed),])
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text=text, reply_markup=markup)
async def set_steps_handler(update, context):
    input_mex = update.message.text
    log.warning(input_mex)
    input_args = input_mex.split('/set_steps ')[1]
    global STEPS
    STEPS=int(input_args)
    await update.message.reply_text(input_args)
async def set_negative_prompt_handler(update, context):
    input_mex = update.message.text
    log.warning(input_mex)
    input_args = input_mex.split('/set_negative_prompt ')[1]
    global NEGATIVE_PROMPT
    NEGATIVE_PROMPT=input_args
    await update.message.reply_text(input_args)
async def set_guidance_scale_handler(update, context):
    input_mex = update.message.text
    log.warning(input_mex)
    input_args = input_mex.split('/set_guidance_scale ')[1]
    global GUIDANCE_SCALE
    GUIDANCE_SCALE=int(input_args)
    await update.message.reply_text(input_args)

app = ApplicationBuilder().token(TG_TOKEN).build()
app.add_handler(CommandHandler('select_model', select_model_handler))
app.add_handler(CommandHandler('select_scheduler', select_scheduler_handler))
app.add_handler(CommandHandler('set_steps', set_steps_handler))
app.add_handler(CommandHandler('set_guidance_scale', set_guidance_scale_handler))
app.add_handler(CommandHandler('set_negative_prompt', set_negative_prompt_handler))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_and_send_photo))
app.add_handler(CallbackQueryHandler(button))
log.warning('Start bot')
app.run_polling()
