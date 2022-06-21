def pytest_addoption(parser):
    # Attaches SHARK command-line arguments to the pytest machinery.
    parser.addoption("--save_mlir", default="False", help="Pass option to save input MLIR module to /tmp/ directory.")
    parser.addoption("--save_vmfb", default="False", help="Pass option to save input MLIR module to /tmp/ directory.")
