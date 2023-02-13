import os
import tempfile
import gradio
from os import listdir

gradio_tmp_imgs_folder = os.getcwd() + "/shark_tmp/"


# Clear all gradio tmp images
def clear_gradio_tmp_imgs_folder():
    if not os.path.exists(gradio_tmp_imgs_folder):
        return
    for fileName in listdir(gradio_tmp_imgs_folder):
        # Delete tmp png files
        if fileName.startswith("tmp") and fileName.endswith(".png"):
            os.remove(gradio_tmp_imgs_folder + fileName)


# Overwrite save_pil_to_file from gradio to save tmp images generated by gradio into our own tmp folder
def save_pil_to_file(pil_image, dir=None):
    if not os.path.exists(gradio_tmp_imgs_folder):
        os.mkdir(gradio_tmp_imgs_folder)
    file_obj = tempfile.NamedTemporaryFile(
        delete=False, suffix=".png", dir=gradio_tmp_imgs_folder
    )
    pil_image.save(file_obj)
    return file_obj


# Register save_pil_to_file override
gradio.processing_utils.save_pil_to_file = save_pil_to_file
