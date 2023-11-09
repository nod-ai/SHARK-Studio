import PIL.Image as Image

resamplers = {
    "Lanczos": Image.Resampling.LANCZOS,
    "Nearest Neighbor": Image.Resampling.NEAREST,
    "Bilinear": Image.Resampling.BILINEAR,
    "Bicubic": Image.Resampling.BICUBIC,
    "Hamming": Image.Resampling.HAMMING,
    "Box": Image.Resampling.BOX,
}

resampler_list = resamplers.keys()
