def pytest_addoption(parser):
    # Attaches SHARK command-line arguments to the pytest machinery.
    parser.addoption(
        "--benchmark",
        action="store_true",
        default="False",
        help="Pass option to benchmark and write results.csv",
    )
    parser.addoption(
        "--onnx_bench",
        action="store_true",
        default="False",
        help="Add ONNX benchmark results to pytest benchmarks.",
    )
    parser.addoption(
        "--tf32",
        action="store_true",
        default="False",
        help="Use TensorFloat-32 calculations.",
    )
    parser.addoption(
        "--save_repro",
        action="store_true",
        default="False",
        help="Pass option to save reproduction artifacts to SHARK/shark_tmp/test_case/",
    )
    parser.addoption(
        "--save_fails",
        action="store_true",
        default="False",
        help="Save reproduction artifacts for a test case only if it fails. Default is False.",
    )
    parser.addoption(
        "--ci",
        action="store_true",
        default="False",
        help="Enables uploading of reproduction artifacts upon test case failure during iree-compile or validation. Must be passed with --ci_sha option ",
    )
    parser.addoption(
        "--ci_sha",
        action="store",
        default="None",
        help="Passes the github SHA of the CI workflow to include in google storage directory for reproduction artifacts.",
    )
    parser.addoption(
        "--tank_url",
        type=str,
        default="gs://shark_tank/latest",
        help="URL to bucket from which to download SHARK tank artifacts. Default is gs://shark_tank/latest",
    )
