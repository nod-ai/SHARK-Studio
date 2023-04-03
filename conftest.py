def pytest_addoption(parser):
    # Attaches SHARK command-line arguments to the pytest machinery.
    parser.addoption(
        "--benchmark",
        action="store",
        type=str,
        default=None,
        choices=("baseline", "native", "all"),
        help="Benchmarks specified engine(s) and writes bench_results.csv.",
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
        "--update_tank",
        action="store_true",
        default="False",
        help="Update local shark tank with latest artifacts if model artifact hash mismatched.",
    )
    parser.addoption(
        "--force_update_tank",
        action="store_true",
        default="False",
        help="Force-update local shark tank with artifacts from specified shark_tank URL (defaults to nightly).",
    )
    parser.addoption(
        "--ci_sha",
        action="store",
        default="None",
        help="Passes the github SHA of the CI workflow to include in google storage directory for reproduction artifacts.",
    )
    parser.addoption(
        "--local_tank_cache",
        action="store",
        default=None,
        help="Specify the directory in which all downloaded shark_tank artifacts will be cached.",
    )
    parser.addoption(
        "--tank_url",
        type=str,
        default="gs://shark_tank/nightly",
        help="URL to bucket from which to download SHARK tank artifacts. Default is gs://shark_tank/latest",
    )
    parser.addoption(
        "--tank_prefix",
        type=str,
        default="nightly",
        help="Prefix to gs://shark_tank/ model directories from which to download SHARK tank artifacts. Default is 'latest'.",
    )
    parser.addoption(
        "--benchmark_dispatches",
        default=None,
        help="Benchmark individual dispatch kernels produced by IREE compiler. Use 'All' for all, or specific dispatches e.g. '0 1 2 10'",
    )
    parser.addoption(
        "--dispatch_benchmarks_dir",
        default="./temp_dispatch_benchmarks",
        help="Directory in which dispatch benchmarks are saved.",
    )
    parser.addoption(
        "--batchsize",
        default=1,
        type=int,
        help="Batch size for the tested model.",
    )
