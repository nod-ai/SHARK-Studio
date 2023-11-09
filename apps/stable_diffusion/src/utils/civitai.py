import re
import requests
from apps.stable_diffusion.src.utils.stable_args import args

from pathlib import Path
from tqdm import tqdm


def get_civitai_checkpoint(url: str):
    with requests.get(url, allow_redirects=True, stream=True) as response:
        response.raise_for_status()

        # civitai api returns the filename in the content disposition
        base_filename = re.findall(
            '"([^"]*)"', response.headers["Content-Disposition"]
        )[0]
        destination_path = (
            Path.cwd() / (args.ckpt_dir or "models") / base_filename
        )

        # we don't have this model downloaded yet
        if not destination_path.is_file():
            print(
                f"downloading civitai model from {url} to {destination_path}"
            )

            size = int(response.headers["content-length"], 0)
            progress_bar = tqdm(total=size, unit="iB", unit_scale=True)

            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

            progress_bar.close()

        # we already have this model downloaded
        else:
            print(f"civitai model already downloaded to {destination_path}")

        response.close()
        return destination_path.as_posix()
