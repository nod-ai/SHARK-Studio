class LanguageModel:
    def __init__(
        self,
        shark_llm_model,
        tokenizer,
    ):
        self.shark_llm_model = shark_llm_model
        self.tokenizer = tokenizer


    def generate_tokens():

    def from_pretrained(model, model_inputs, model_name, device, precision):
        from shark.shark_inference import SharkInference

        # device = "cuda"  # "cpu"
        # TODO: vmfb and mlir name should include precision and device
        vmfb_path = (
            Path(model_name + f"_{device}.vmfb")
            if model_vmfb_name is None
            else Path(model_vmfb_name)
        )
        shark_module = get_vmfb_from_path(
            vmfb_path, device, mlir_dialect="tm_tensor"
        )
        if shark_module is not None:
            return shark_module

        mlir_path = Path(model_name + ".mlir")
        print(
            f"[DEBUG] mlir path {mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
        )
        if mlir_path.exists():
            with open(mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            ts_graph = get_torch_mlir_module_bytecode(model, model_inputs)
            module = torch_mlir.compile(
                ts_graph,
                [*model_inputs],
                torch_mlir.OutputType.LINALG_ON_TENSORS,
                use_tracing=False,
                verbose=False,
            )
            bytecode_stream = BytesIO()
            module.operation.write_bytecode(bytecode_stream)
            bytecode = bytecode_stream.getvalue()
        f_ = open(model_name + ".mlir", "wb")
        f_.write(bytecode)
        print("Saved mlir")
        f_.close()

        shark_module = SharkInference(
            mlir_module=bytecode, device=device, mlir_dialect="tm_tensor"
        )
        shark_module.compile()

        path = shark_module.save_module(
            vmfb_path.parent.absolute(), vmfb_path.stem
        )
        print("Saved vmfb at ", str(path))

        return shark_module




