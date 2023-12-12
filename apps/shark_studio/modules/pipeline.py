from shark.iree_utils.compile_utils import get_iree_compiled_module


class SharkPipelineBase:
    # This class is a lightweight base for managing an
    # inference API class. It should provide methods for:
    # - compiling a set (model map) of torch IR modules
    # - preparing weights for an inference job
    # - loading weights for an inference job
    # - utilites like benchmarks, tests

    def __init__(
        self,
        model_map: dict,
        device: str,
        import_mlir: bool = True,
    ):
        self.model_map = model_map
        self.device = device
        self.import_mlir = import_mlir

    def import_torch_ir(self, base_model_id):
        for submodel in self.model_map:
            hf_id = (
                submodel["custom_hf_id"]
                if submodel["custom_hf_id"]
                else base_model_id
            )
            torch_ir = submodel["initializer"](
                hf_id, **submodel["init_kwargs"], compile_to="torch"
            )
            submodel["tempfile_name"] = get_resource_path(
                f"{submodel}.torch.tempfile"
            )
            with open(submodel["tempfile_name"], "w+") as f:
                f.write(torch_ir)
            del torch_ir
            gc.collect()

    def load_vmfb(self, submodel):
        if self.iree_module_dict[submodel]:
            print(
                f".vmfb for {submodel} found at {self.iree_module_dict[submodel]['vmfb']}"
            )
        elif self.model_map[submodel]["tempfile_name"]:
            submodel["tempfile_name"]

        return submodel["vmfb"]

    def merge_custom_map(self, custom_model_map):
        for submodel in custom_model_map:
            for key in submodel:
                self.model_map[submodel][key] = key
        print(self.model_map)

    def get_compiled_map(self, device) -> None:
        # this comes with keys: "vmfb", "config", and "temp_file_to_unlink".
        for submodel in self.model_map:
            if not self.iree_module_dict[submodel][vmfb]:
                self.iree_module_dict[submodel] = get_iree_compiled_module(
                    submodel.tempfile_name,
                    device=self.device,
                    frontend="torch",
                )
        # TODO: delete the temp file

    def run(self, submodel, inputs):
        return

    def safe_name(name):
        return name.replace("/", "_").replace("-", "_")
