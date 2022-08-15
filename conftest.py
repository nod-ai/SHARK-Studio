def pytest_addoption(parser):
    # Attaches SHARK command-line arguments to the pytest machinery.
    parser.addoption(
        "--benchmark",
        action="store_true",
        default="False",
        help="Pass option to benchmark and write results.csv",
    )
    parser.addoption(
        "--onnx_bench"
        action="store_true",
        default="False",
        help="Add ONNX benchmark results to pytest benchmarks."
    )
    # The following options are deprecated and pending removal.
    parser.addoption(
        "--save_mlir",
        action="store_true",
        default="False",
        help="Pass option to save input MLIR",
    )
    parser.addoption(
        "--save_vmfb",
        action="store_true",
        default="False",
        help="Pass option to save IREE output .vmfb",
    )
    parser.addoption(
        "--save_temps",
        action="store_true",
        default="False",
        help="Saves IREE reproduction artifacts for filing upstream issues.",
    )
