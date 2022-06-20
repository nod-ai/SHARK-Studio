def pytest_addoption(parser):
    # Attaches SHARK command-line arguments to the pytest machinery.
    parser.addoption("--save_temps", action="store_true", default="False", help="Saves IREE reproduction artifacts for filing upstream issues.")
