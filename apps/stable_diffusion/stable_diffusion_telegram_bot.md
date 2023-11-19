1. You need to create your bot here https://core.telegram.org/bots#how-do-i-create-a-bot
2. Then create in the directory ./apps/stable_diffusion/scripts .env file eith content
TG_TOKEN="your_token" - specifying your bot's token from previous step.
3. In shark.venv run pip install -r requirements-tgbot.txt   
4. Then run telegram_bot.py with the same parameters that you use when running index.py, for example:
 python .\apps\stable_diffusion\scripts\telegram_bot.py --local_tank_cache=H:\SHARK\TEMP --use_base_vae --vulkan_large_heap_block_size=0

Bot commands:
/select_model
/select_scheduler
/set_steps "integer number of steps"
/set_guidance_scale "integer number"
/set_negative_prompt "negative text"

Any other text create an image based on it.
