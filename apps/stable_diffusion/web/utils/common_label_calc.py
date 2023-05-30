# functions for generating labels used in common by tabs across the UI


def status_label(tab_name, batch_index=0, batch_count=1, batch_size=1):
    if batch_index < batch_count:
        bs = f"x{batch_size}" if batch_size > 1 else ""
        return f"{tab_name} generating {batch_index+1}/{batch_count}{bs}"
    else:
        return f"{tab_name} complete"
