import glob
import gradio as gr
import os
import subprocess
import sys
from PIL import Image

from apps.stable_diffusion.src import args
from apps.stable_diffusion.src.utils import (
    get_generated_imgs_path,
    get_generated_imgs_todays_subdir,
)
from apps.stable_diffusion.web.ui.utils import nodlogo_loc
from apps.stable_diffusion.web.utils.metadata import displayable_metadata

# -- Functions for file, directory and image info querying

output_dir = get_generated_imgs_path()


def outputgallery_filenames(subdir) -> list[str]:
    new_dir_path = os.path.join(output_dir, subdir)
    if os.path.exists(new_dir_path):
        filenames = [
            glob.glob(new_dir_path + "/" + ext)
            for ext in ("*.png", "*.jpg", "*.jpeg")
        ]

        return sorted(sum(filenames, []), key=os.path.getmtime, reverse=True)
    else:
        return []


def output_subdirs() -> list[str]:
    # Gets a list of subdirectories of output_dir and below, as relative paths.
    relative_paths = [
        os.path.relpath(entry[0], output_dir)
        for entry in os.walk(
            output_dir, followlinks=args.output_gallery_followlinks
        )
    ]

    # It is less confusing to always including the subdir that will take any
    # images generated today even if it doesn't exist yet
    if get_generated_imgs_todays_subdir() not in relative_paths:
        relative_paths.append(get_generated_imgs_todays_subdir())

    # sort subdirectories so that the date named ones we probably
    # created in this or previous sessions come first, sorted with the most
    # recent first. Other subdirs are listed after.
    generated_paths = sorted(
        [path for path in relative_paths if path.isnumeric()], reverse=True
    )
    result_paths = generated_paths + sorted(
        [
            path
            for path in relative_paths
            if (not path.isnumeric()) and path != "."
        ]
    )

    return result_paths


# --- Define UI layout for Gradio

with gr.Blocks() as outputgallery_web:
    nod_logo = Image.open(nodlogo_loc)

    with gr.Row(elem_id="outputgallery_gallery"):
        # needed to workaround gradio issue:
        # https://github.com/gradio-app/gradio/issues/2907
        dev_null = gr.Textbox("", visible=False)

        gallery_files = gr.State(value=[])
        subdirectory_paths = gr.State(value=[])

        with gr.Column(scale=6):
            logo = gr.Image(
                label="Getting subdirectories...",
                value=nod_logo,
                interactive=False,
                visible=True,
                show_label=True,
                elem_id="top_logo",
                elem_classes="logo_centered",
            )

            gallery = gr.Gallery(
                label="",
                value=gallery_files.value,
                visible=False,
                show_label=True,
            ).style(columns=4)

        with gr.Column(scale=4):
            with gr.Box():
                with gr.Row():
                    with gr.Column(scale=15, min_width=160):
                        subdirectories = gr.Dropdown(
                            label=f"Subdirectories of {output_dir}",
                            type="value",
                            choices=subdirectory_paths.value,
                            value="",
                            interactive=True,
                        ).style(container=False)
                    with gr.Column(
                        scale=1,
                        min_width=32,
                        elem_classes="output_icon_button",
                    ):
                        open_subdir = gr.Button(
                            variant="secondary",
                            value="\U0001F5C1",  # unicode open folder
                            interactive=False,
                        ).style(size="sm")
                    with gr.Column(
                        scale=1,
                        min_width=32,
                        elem_classes="output_icon_button",
                    ):
                        refresh = gr.Button(
                            variant="secondary",
                            value="\u21BB",  # unicode clockwise arrow circle
                        ).style(size="sm")

            image_columns = gr.Slider(
                label="Columns shown", value=4, minimum=1, maximum=16, step=1
            )
            outputgallery_filename = gr.Textbox(
                label="Filename", value="None", interactive=False
            ).style(show_copy_button=True)

            with gr.Accordion(
                label="Parameter Information", open=False
            ) as parameters_accordian:
                image_parameters = gr.DataFrame(
                    headers=["Parameter", "Value"],
                    col_count=2,
                    wrap=True,
                    elem_classes="output_parameters_dataframe",
                    value=[["Status", "No image selected"]],
                )

            with gr.Accordion(label="Send To", open=True):
                with gr.Row():
                    outputgallery_sendto_txt2img = gr.Button(
                        value="Txt2Img",
                        interactive=False,
                        elem_classes="outputgallery_sendto",
                    ).style(size="sm")

                    outputgallery_sendto_img2img = gr.Button(
                        value="Img2Img",
                        interactive=False,
                        elem_classes="outputgallery_sendto",
                    ).style(size="sm")

                    outputgallery_sendto_inpaint = gr.Button(
                        value="Inpaint",
                        interactive=False,
                        elem_classes="outputgallery_sendto",
                    ).style(size="sm")

                    outputgallery_sendto_outpaint = gr.Button(
                        value="Outpaint",
                        interactive=False,
                        elem_classes="outputgallery_sendto",
                    ).style(size="sm")

                    outputgallery_sendto_upscaler = gr.Button(
                        value="Upscaler",
                        interactive=False,
                        elem_classes="outputgallery_sendto",
                    ).style(size="sm")

    # --- Event handlers

    def on_clear_gallery():
        return [
            gr.Gallery.update(
                value=[],
                visible=False,
            ),
            gr.Image.update(
                visible=True,
            ),
        ]

    def on_select_subdir(subdir) -> list:
        # evt.value is the subdirectory name
        new_images = outputgallery_filenames(subdir)
        new_label = (
            f"{len(new_images)} images in {os.path.join(output_dir, subdir)}"
        )
        return [
            new_images,
            gr.Gallery.update(
                value=new_images,
                label=new_label,
                visible=len(new_images) > 0,
            ),
            gr.Image.update(
                label=new_label,
                visible=len(new_images) == 0,
            ),
        ]

    def on_open_subdir(subdir):
        subdir_path = os.path.normpath(os.path.join(output_dir, subdir))

        if os.path.isdir(subdir_path):
            if sys.platform == "linux":
                subprocess.run(["xdg-open", subdir_path])
            elif sys.platform == "darwin":
                subprocess.run(["open", subdir_path])
            elif sys.platform == "win32":
                os.startfile(subdir_path)

    def on_refresh(current_subdir: str) -> list:
        # get an up-to-date subdirectory list
        refreshed_subdirs = output_subdirs()
        # get the images using either the current subdirectory or the most
        # recent valid one
        new_subdir = (
            current_subdir
            if current_subdir in refreshed_subdirs
            else refreshed_subdirs[0]
        )
        new_images = outputgallery_filenames(new_subdir)
        new_label = (
            f"{len(new_images)} images in "
            f"{os.path.join(output_dir, new_subdir)}"
        )

        return [
            gr.Dropdown.update(
                choices=refreshed_subdirs,
                value=new_subdir,
            ),
            refreshed_subdirs,
            new_images,
            gr.Gallery.update(
                value=new_images, label=new_label, visible=len(new_images) > 0
            ),
            gr.Image.update(
                label=new_label,
                visible=len(new_images) == 0,
            ),
        ]

    def on_new_image(subdir, subdir_paths, status) -> list:
        # prevent error triggered when an image generates before the tab
        # has even been selected
        subdir_paths = (
            subdir_paths
            if len(subdir_paths) > 0
            else [get_generated_imgs_todays_subdir()]
        )

        # only update if the current subdir is the most recent one as
        # new images only go there
        if subdir_paths[0] == subdir:
            new_images = outputgallery_filenames(subdir)
            new_label = (
                f"{len(new_images)} images in "
                f"{os.path.join(output_dir, subdir)} - {status}"
            )

            return [
                new_images,
                gr.Gallery.update(
                    value=new_images,
                    label=new_label,
                    visible=len(new_images) > 0,
                ),
                gr.Image.update(
                    label=new_label,
                    visible=len(new_images) == 0,
                ),
            ]
        else:
            # otherwise change nothing,
            # (only untyped gradio gr.update() does this)
            return [gr.update(), gr.update(), gr.update()]

    def on_select_image(images: list[str], evt: gr.SelectData) -> list:
        # evt.index is an index into the full list of filenames for
        # the current subdirectory
        filename = images[evt.index]
        params = displayable_metadata(filename)

        if params:
            if params["source"] == "missing":
                return [
                    "Could not find this image file, refresh the gallery and update the images",
                    [["Status", "File missing"]],
                ]
            else:
                return [
                    filename,
                    list(map(list, params["parameters"].items())),
                ]

        return [
            filename,
            [["Status", "No parameters found"]],
        ]

    def on_outputgallery_filename_change(filename: str) -> list:
        exists = filename != "None" and os.path.exists(filename)
        return [
            # disable or enable each of the sendto button based on whether
            # an image is selected
            gr.Button.update(interactive=exists),
            gr.Button.update(interactive=exists),
            gr.Button.update(interactive=exists),
            gr.Button.update(interactive=exists),
            gr.Button.update(interactive=exists),
            gr.Button.update(interactive=exists),
        ]

    # The time first our tab is selected we need to do an initial refresh
    # to populate the subdirectory select box and the images from the most
    # recent subdirectory.
    #
    # We do it at this point rather than setting this up in the controls'
    # definitions as when you refresh the browser you always get what was
    # *initially* set, which won't include any new subdirectories or images
    # that might have created since the application was started. Doing it
    # this way means a browser refresh/reload always gets the most
    # up-to-date data.
    def on_select_tab(subdir_paths, request: gr.Request):
        local_client = request.headers["host"].startswith(
            "127.0.0.1:"
        ) or request.headers["host"].startswith("localhost:")

        if len(subdir_paths) == 0:
            return on_refresh("") + [gr.update(interactive=local_client)]
        else:
            return (
                # Change nothing, (only untyped gr.update() does this)
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

    # Unfortunately as of gradio 3.34.0 gr.update against Galleries doesn't
    # support things set with .style, nor the elem_classes kwarg, so we have
    # to directly set things up via JavaScript if we want the client to take
    # notice of our changes to the number of columns after it decides to put
    # them back to the original number when we change something
    def js_set_columns_in_browser(timeout_length):
        return f"""
            (new_cols) => {{
                setTimeout(() => {{
                    required_style = "auto ".repeat(new_cols).trim();
                    gallery = document.querySelector('#outputgallery_gallery .grid-container');
                    if (gallery) {{
                        gallery.style.gridTemplateColumns = required_style
                    }}
                }}, {timeout_length});
                return [];      // prevents console error from gradio
            }}
        """

    # --- Wire handlers up to the actions

    # Many actions reset the number of columns shown in the gallery on the
    # browser end, so we have to set them back to what we think they should
    # be after the initial action.
    #
    # None of the actions on this tab trigger inference, and we want the
    # user to be able to do them whilst other tabs have ongoing inference
    # running. Waiting in the queue behind inference jobs would mean the UI
    # can't fully respond until the inference tasks complete,
    # hence queue=False on all of these.
    set_gallery_columns_immediate = dict(
        fn=None,
        inputs=[image_columns],
        # gradio blanks the UI on Chrome on Linux on gallery select if
        # I don't put an output here
        outputs=[dev_null],
        _js=js_set_columns_in_browser(0),
        queue=False,
    )

    # setting columns after selecting a gallery item needs a real
    # timeout length for the number of columns to actually be applied.
    # Not really sure why, maybe something has to finish animating?
    set_gallery_columns_delayed = dict(
        set_gallery_columns_immediate, _js=js_set_columns_in_browser(250)
    )

    # clearing images when we need to completely change what's in the
    # gallery avoids current images being shown replacing piecemeal and
    # prevents weirdness and errors if the user selects an image during the
    # replacement phase.
    clear_gallery = dict(
        fn=on_clear_gallery,
        inputs=None,
        outputs=[gallery, logo],
        queue=False,
    )

    image_columns.change(**set_gallery_columns_immediate)

    subdirectories.select(**clear_gallery).then(
        on_select_subdir,
        [subdirectories],
        [gallery_files, gallery, logo],
        queue=False,
    ).then(**set_gallery_columns_immediate)

    open_subdir.click(
        on_open_subdir, inputs=[subdirectories], queue=False
    ).then(**set_gallery_columns_immediate)

    refresh.click(**clear_gallery).then(
        on_refresh,
        [subdirectories],
        [subdirectories, subdirectory_paths, gallery_files, gallery, logo],
        queue=False,
    ).then(**set_gallery_columns_immediate)

    gallery.select(
        on_select_image,
        [gallery_files],
        [outputgallery_filename, image_parameters],
        queue=False,
    ).then(**set_gallery_columns_delayed)

    outputgallery_filename.change(
        on_outputgallery_filename_change,
        [outputgallery_filename],
        [
            outputgallery_sendto_txt2img,
            outputgallery_sendto_img2img,
            outputgallery_sendto_inpaint,
            outputgallery_sendto_outpaint,
            outputgallery_sendto_upscaler,
        ],
        queue=False,
    )

    # We should have been given the .select function for our tab, so set it up
    def outputgallery_tab_select(select):
        select(
            fn=on_select_tab,
            inputs=[subdirectory_paths],
            outputs=[
                subdirectories,
                subdirectory_paths,
                gallery_files,
                gallery,
                logo,
                open_subdir,
            ],
            queue=False,
        ).then(**set_gallery_columns_immediate)

    # We should have been passed a list of components on other tabs that update
    # when a new image has generated on that tab, so set things up so the user
    # will see that new image if they are looking at today's subdirectory
    def outputgallery_watch(components: gr.Textbox):
        for component in components:
            component.change(
                on_new_image,
                inputs=[subdirectories, subdirectory_paths, component],
                outputs=[gallery_files, gallery, logo],
                queue=False,
            ).then(**set_gallery_columns_immediate)
