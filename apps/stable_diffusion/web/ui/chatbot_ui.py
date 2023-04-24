from fastchat import build_demo, get_model_list, set_global_vars

controller_url = "http://localhost:21001"
moderate = False
models = get_model_list(controller_url)
set_global_vars(controller_url, moderate, models)

chatbot_web = build_demo()
