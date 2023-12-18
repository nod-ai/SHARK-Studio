import apps.shark_studio.web.utils.globals as global_obj
import gc


def status_label(tab_name, batch_index=0, batch_count=1, batch_size=1):
    print(f"Getting status label for {tab_name}")
    if batch_index < batch_count:
        bs = f"x{batch_size}" if batch_size > 1 else ""
        return f"{tab_name} generating {batch_index+1}/{batch_count}{bs}"
    else:
        return f"{tab_name} complete"


def get_generation_text_info(seeds, device):
    cfg_dump = {}
    for cfg in global_obj.get_config_dict():
        cfg_dump[cfg] = cfg
    text_output = f"prompt={cfg_dump['prompts']}"
    text_output += f"\nnegative prompt={cfg_dump['negative_prompts']}"
    text_output += (
        f"\nmodel_id={cfg_dump['hf_model_id']}, " f"ckpt_loc={cfg_dump['ckpt_loc']}"
    )
    text_output += f"\nscheduler={cfg_dump['scheduler']}, " f"device={device}"
    text_output += (
        f"\nsteps={cfg_dump['steps']}, "
        f"guidance_scale={cfg_dump['guidance_scale']}, "
        f"seed={seeds}"
    )
    text_output += (
        f"\nsize={cfg_dump['height']}x{cfg_dump['width']}, "
        if not cfg_dump.use_hiresfix
        else f"\nsize={cfg_dump['hiresfix_height']}x{cfg_dump['hiresfix_width']}, "
    )
    text_output += (
        f"batch_count={cfg_dump['batch_count']}, "
        f"batch_size={cfg_dump['batch_size']}, "
        f"max_length={cfg_dump['max_length']}"
    )

    return text_output
