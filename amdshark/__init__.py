import importlib
import logging

from torch._dynamo import register_backend

log = logging.getLogger(__name__)


@register_backend
def amdshark(model, inputs, *, options):
    try:
        from amdshark.dynamo_backend.utils import AMDSharkBackend
    except ImportError:
        log.exception(
            "Unable to import AMDSHARK - High Performance Machine Learning Distribution"
            "Please install the right version of AMDSHARK that matches the PyTorch version being used. "
            "Refer to https://github.com/nod-ai/AMDSHARK-Studio/ for details."
        )
        raise
    return AMDSharkBackend(model, inputs, options)


def has_amdshark():
    try:
        importlib.import_module("amdshark")
        return True
    except ImportError:
        return False
