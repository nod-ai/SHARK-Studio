import gradio.routes
import mimetypes
import tempfile
import os
from apps.stable_diffusion.src import args

# Fix for Windows users.
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")


class ScriptLoader:
    path_map = {
        "js": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "javascript")
        ),
    }

    def __init__(self, script_type):
        self.script_type = script_type
        self.path = ScriptLoader.path_map[script_type]
        self.loaded_scripts = []

    @staticmethod
    def get_scripts(path: str, file_type: str) -> list[tuple[str, str]]:
        """Returns list of tuples
        Each tuple contains the full filepath and filename as strings
        """
        scripts = []
        dir_list = [os.path.join(path, f) for f in os.listdir(path)]
        files_list = [f for f in dir_list if os.path.isfile(f)]
        for s in files_list:
            # Dont forget the "." for file extension
            if os.path.splitext(s)[1] == f".{file_type}":
                scripts.append((s, os.path.basename(s)))

        return scripts


class JavaScriptLoader(ScriptLoader):
    def __init__(self):
        # Script type set here
        super().__init__("js")
        # Copy the template response
        self.original_template = gradio.routes.templates.TemplateResponse
        # Load the js files
        self.load_js()
        # Init the selected Gradio template
        self.init_template()
        # reassign the template response to your method, so gradio calls your method instead
        gradio.routes.templates.TemplateResponse = self.template_response

    def load_js(self):
        js_scripts = ScriptLoader.get_scripts(self.path, self.script_type)
        for file_path, file_name in js_scripts:
            with open(file_path, "r", encoding="utf-8") as file:
                self.loaded_scripts.append(
                    f"\n<!--{file_name}-->\n<script>\n{file.read()}\n</script>"
                )

    def init_template(self):
        if args.theme != "light":
            self.loaded_scripts += f'<script type="text/javascript">set_theme("{args.theme}")</script>\n'

    def template_response(self, *args, **kwargs):
        """Once gradio calls your method, you call the original, you modify it to include
        your scripts and you return the modified version
        """
        response = self.original_template(*args, **kwargs)
        response.body = response.body.replace(
            "</head>".encode("utf-8"),
            f"{''.join(self.loaded_scripts)}\n</head>".encode("utf-8"),
        )
        response.init_headers()
        return response
