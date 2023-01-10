You need to pre-create your bot (https://core.telegram.org/bots#how-do-i-create-a-bot)
Then create in the directory web file .env
In it the record:
TG_TOKEN="your_token"
specifying your bot's token from previous step.
Then run bot.py with the same parameters that you use when running index.py, for example:
python bot.py --max_length=77 --vulkan_large_heap_block_size=0 --use_base_vae --local_tank_cache h:\shark\TEMP

Bot commands:
/select_model
/select_scheduler
/set_steps "integer number of steps"
/set_guidance_scale "integer number"
/set_negative_prompt "negative text"
Any other text triggers the creation of an image based on it.
