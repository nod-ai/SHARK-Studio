import gc

"""
The global objects include SD pipeline and config.
Maintaining the global objects would avoid creating extra pipeline objects when switching modes.
Also we could avoid memory leak when switching models by clearing the cache.
"""


def _init():
    global _sd_obj
    global _pipe_kwargs
    global _gen_kwargs
    global _schedulers
    _sd_obj = None
    _pipe_kwargs = None
    _gen_kwargs = None
    _schedulers = None


def set_sd_obj(value):
    global _sd_obj
    _sd_obj = value


def set_sd_scheduler(key):
    global _sd_obj
    _sd_obj.scheduler = _schedulers[key]


def set_sd_status(value):
    global _sd_obj
    _sd_obj.status = value


def set_pipe_kwargs(value):
    global _pipe_kwargs
    _pipe_kwargs = value


def set_gen_kwargs(value):
    global _gen_kwargs
    _gen_kwargs = value


def set_schedulers(value):
    global _schedulers
    _schedulers = value


def get_sd_obj():
    global _sd_obj
    return _sd_obj


def get_sd_status():
    global _sd_obj
    return _sd_obj.status


def get_pipe_kwargs():
    global _pipe_kwargs
    return _pipe_kwargs


def get_gen_kwargs():
    global _gen_kwargs
    return _gen_kwargs


def get_scheduler(key):
    global _schedulers
    return _schedulers[key]


def clear_cache():
    global _sd_obj
    global _pipe_kwargs
    global _gen_kwargs
    global _schedulers
    del _sd_obj
    del _pipe_kwargs
    del _gen_kwargs
    del _schedulers
    gc.collect()
    _sd_obj = None
    _pipe_kwargs = None
    _gen_kwargs = None
    _schedulers = None
