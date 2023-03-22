import gc


"""
The global objects include SD pipeline and config.
Maintaining the global objects would avoid creating extra pipeline objects when switching modes.
Also we could avoid memory leak when switching models by clearing the cache.
"""


def _init():
    global _sd_obj
    global _config_obj
    _sd_obj = None
    _config_obj = None


def set_sd_obj(value):
    global _sd_obj
    _sd_obj = value


def set_schedulers(value):
    global _sd_obj
    _sd_obj.scheduler = value


def set_sd_status(value):
    global _sd_obj
    _sd_obj.status = value


def set_cfg_obj(value):
    global _config_obj
    _config_obj = value


def get_sd_obj():
    return _sd_obj


def get_sd_status():
    return _sd_obj.status


def get_cfg_obj():
    return _config_obj


def clear_cache():
    global _sd_obj
    global _config_obj
    del _sd_obj
    del _config_obj
    gc.collect()
    _sd_obj = None
    _config_obj = None
