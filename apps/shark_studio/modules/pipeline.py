from shark.iree_utils.compile_utils import get_iree_compiled_module
from apps.shark_studio.web.utils.file_utils import get_checkpoints_path
import gc


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
        base_model_id: str,
        device: str,
        import_mlir: bool = True,
    ):
        self.model_map = model_map
        self.base_model_id = base_model_id
        self.device = device
        self.import_mlir = import_mlir
        self.iree_module_dict = {}


    def import_torch_ir(self, submodel, kwargs):
        weights = (
            submodel["custom_weights"]
            if submodel["custom_weights"]
            else None
        )
        torch_ir = self.model_map[submodel]["initializer"](
            self.base_model_id, **kwargs, compile_to="torch"
        )
        self.model_map[submodel]["tempfile_name"] = get_resource_path(
            f"{submodel}.torch.tempfile"
        )
        with open(self.model_map[submodel]["tempfile_name"], "w+") as f:
            f.write(torch_ir)
        del torch_ir
        gc.collect()


    def load_vmfb(self, submodel):
        if submodel in self.iree_module_dict:
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


    def get_local_vmfbs(self, pipe_id):
        for submodel in self.model_map:
            vmfbs = []
            vmfb_matches = {}
            vmfbs_path = get_checkpoints_path("../vmfbs")
            for (dirpath, dirnames, filenames) in os.walk(vmfbs_path):
                vmfbs.extend(filenames)
                break
            for file in vmfbs:
                if all(keys in file for keys in [submodel, pipe_id]):
                    print(f"Found existing .vmfb at {file}")
                    self.iree_module_dict[submodel] = {'vmfb': file}
            

    def get_compiled_map(self, device, pipe_id) -> None:
        # this comes with keys: "vmfb", "config", and "temp_file_to_unlink".
        if not self.import_mlir:
            self.get_local_vmfbs(pipe_id)
        for submodel in self.model_map:
            if submodel in self.iree_module_dict:
                if "vmfb" in self.iree_module_dict[submodel]:
                    continue
                if "tempfile_name" not in self.model_map[submodel]:
                    sub_kwargs = self.model_map[submodel]["kwargs"] if self.model_map[submodel]["kwargs"] else {}
                    import_torch_ir(submodel, self.base_model_id, **sub_kwargs)
                self.iree_module_dict[submodel] = get_iree_compiled_module(
                    submodel["tempfile_name"],
                    device=self.device,
                    frontend="torch",
                    external_weight_file=submodel["custom_weights"]
                )
        # TODO: delete the temp file


    def run(self, submodel, inputs):
        return


    def safe_name(name):
        return name.replace("/", "_").replace("-", "_")
