def pytest_addoption(parser):
    # Attaches SHARK command-line arguments to the pytest machinery.
    parser.addoption("--save_mlir", action="store_true", default="False", help="Pass option to save input MLIR module to /tmp/ directory.")
