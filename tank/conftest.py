def pytest_addoption(parser):
    # Attaches SHARK command-line arguments to the pytest machinery.
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
        "--benchmark",
        action="store_true",
        default="False",
        help="Pass option to benchmark and write results.csv",
    )
    parser.addoption(
        "--save_temps",
        action="store_true",
        default="False",
        help="Saves IREE reproduction artifacts for filing upstream issues.",
    )
