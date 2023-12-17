from msvcrt import kbhit
from shark.iree_utils.compile_utils import get_iree_compiled_module, load_vmfb_using_mmap
from apps.shark_studio.web.utils.file_utils import (
    get_checkpoints_path,
    get_resource_path,
)
from apps.shark_studio.modules.shared_cmd_opts import (
    cmd_opts,
)
from iree import runtime as ireert
from pathlib import Path
import gc
import os


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
        static_kwargs: dict,
        device: str,
        import_mlir: bool = True,
    ):
        self.model_map = model_map
        self.static_kwargs = static_kwargs
        self.base_model_id = base_model_id
        self.device = device
        self.import_mlir = import_mlir
        self.iree_module_dict = {}
        self.tempfiles = {}


    def get_compiled_map(self, pipe_id, submodel="None", init_kwargs={}) -> None:
        # First checks whether we have .vmfbs precompiled, then populates the map
        # with the precompiled executables and fetches executables for the rest of the map.
        # The weights aren't static here anymore so this function should be a part of pipeline
        # initialization. As soon as you have a pipeline ID unique to your static torch IR parameters,
        # and your model map is populated with any IR - unique model IDs and their static params,
        # call this method to get the artifacts associated with your map.
        self.pipe_id = pipe_id
        self.pipe_vmfb_path = Path(os.path.join(get_checkpoints_path(".."), self.pipe_id))
        self.pipe_vmfb_path.mkdir(parents=True, exist_ok=True)
        print("\n[LOG] Checking for pre-compiled artifacts.")
        if submodel == "None":
            for key in self.model_map:
                self.get_compiled_map(pipe_id, submodel=key)
        else:  
            self.get_precompiled(pipe_id, submodel)
            ireec_flags = []
            if submodel in self.iree_module_dict:
                if "vmfb" in self.iree_module_dict[submodel]:
                    print(f"[LOG] Found executable for {submodel} at {self.iree_module_dict[submodel]['vmfb']}...")
                    return
            elif submodel not in self.tempfiles:
                print(f"[LOG] Tempfile for {submodel} not found. Fetching torch IR...")
                if submodel in self.static_kwargs:
                    init_kwargs = self.static_kwargs[submodel]
                for key in self.static_kwargs["pipe"]:
                    if key not in init_kwargs:
                        init_kwargs[key] = self.static_kwargs["pipe"][key]
                self.import_torch_ir(
                    submodel, init_kwargs
                )
                self.get_compiled_map(pipe_id, submodel)
            else:            
                ireec_flags = self.model_map[submodel]["ireec_flags"] if "ireec_flags" in self.model_map[submodel] else []

                if "external_weights_file" in self.model_map[submodel]:
                    weights_path = self.model_map[submodel]["external_weights_file"]
                else:
                    weights_path = None
                self.iree_module_dict[submodel] = get_iree_compiled_module(
                    self.tempfiles[submodel],
                    device=self.device,
                    frontend="torch",
                    mmap=True,
                    external_weight_file=weights_path,
                    extra_args=ireec_flags,
                    write_to=os.path.join(self.pipe_vmfb_path, submodel + ".vmfb")
                )
        return


    def hijack_weights(self, weights_path, submodel="None"):
        if submodel == "None":
            for i in self.model_map:
                self.hijack_weights(weights_path, i)
        else:
            if submodel in self.iree_module_dict:
                self.model_map[submodel]["external_weights_file"] = weights_path
        return


    def get_precompiled(self, pipe_id, submodel="None"):
        if submodel == "None":
            for model in self.model_map:
                self.get_precompiled(pipe_id, model)
        vmfbs = []
        vmfb_matches = {}
        vmfbs_path = self.pipe_vmfb_path
        for dirpath, dirnames, filenames in os.walk(vmfbs_path):
            vmfbs.extend(filenames)
            break
        for file in vmfbs:
            if submodel in file:
                print(f"Found existing .vmfb at {file}")
                self.iree_module_dict[submodel] = {}
                (
                    self.iree_module_dict[submodel]["vmfb"],
                    self.iree_module_dict[submodel]["config"],
                    self.iree_module_dict[submodel]["temp_file_to_unlink"],
                ) = load_vmfb_using_mmap(
                    os.path.join(vmfbs_path, file),
                    self.device,
                    device_idx=0,
                    rt_flags=[],
                    external_weight_file=self.model_map[submodel]['external_weight_file'],
                )
        return


    def safe_dict(self, kwargs: dict):
        flat_args = {}
        for i in kwargs:
            if isinstance(kwargs[i], dict) and "pass_dict" not in kwargs[i]:
                flat_args[i] = [kwargs[i][j] for j in kwargs[i]]
            else:
                flat_args[i] = kwargs[i]

        return flat_args   


    def import_torch_ir(self, submodel, kwargs):
        torch_ir = self.model_map[submodel]["initializer"](
            **self.safe_dict(kwargs), compile_to="torch"
        )
        if submodel == "clip":
            # clip.export_clip_model returns (torch_ir, tokenizer)
            torch_ir = torch_ir[0]
        self.tempfiles[submodel] = get_resource_path(os.path.join(
            "..", "shark_tmp", f"{submodel}.torch.tempfile"
        ))
        
        with open(self.tempfiles[submodel], "w+") as f:
            f.write(torch_ir)
        del torch_ir
        gc.collect()
        return


    def load_submodels(self, submodels: list):
        for submodel in submodels:
            if submodel in self.iree_module_dict:
                print(
                    f"\n[LOG] Loading .vmfb for {submodel} from {self.iree_module_dict[submodel]['vmfb']}"
                )
            else:
                self.get_compiled_map(self.pipe_id, submodel)
        return


    def run(self, submodel, inputs):
        inp = [ireert.asdevicearray(self.iree_module_dict[submodel]["config"].device, inputs)]
        return self.iree_module_dict[submodel]['vmfb']['main'](*inp)


    def safe_name(name):
        return name.replace("/", "_").replace("-", "_")
