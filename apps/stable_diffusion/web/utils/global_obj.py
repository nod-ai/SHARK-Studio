import gc


"""
The global objects include SD pipeline and config.
Maintaining the global objects would avoid creating extra pipeline objects when switching modes.
Also we could avoid memory leak when switching models by clearing the cache.
"""


def init():
    global sd_obj
    global config_obj
    sd_obj = None
    config_obj = None


def set_sd_obj(value):
    global sd_obj
    sd_obj = value


def set_cfg_obj(value):
    global config_obj
    config_obj = value


def set_schedulers(value):
    global sd_obj
    sd_obj.scheduler = value


def get_sd_obj():
    return sd_obj


def get_cfg_obj():
    return config_obj


def set_sd_status(value):
    global sd_obj
    sd_obj.status = value


def get_sd_status():
    global sd_obj
    return sd_obj.status


def clear_cache():
    global sd_obj
    global config_obj
    del sd_obj
    del config_obj
    gc.collect()
