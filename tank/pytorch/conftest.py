def pytest_addoption(parser):
    # Attaches SHARK command-line arguments to the pytest machinery.
    parser.addoption("--save_mlir", action="store_true", help="Pass option to save input MLIR module to /tmp/ directory.")
