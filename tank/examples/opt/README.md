# Run OPT for sentence completion through SHARK

From base SHARK directory, follow instructions to set up a virtual environment with SHARK. (`./setup_venv.sh` or `./setup_venv.ps1`)
Then, you may run opt_causallm.py to get a very simple sentence completion application running through SHARK
```
python opt_causallm.py
```

# Run OPT performance comparison on SHARK vs. PyTorch

```
python opt_perf_comparison.py --max-seq-len=512 --model-name=facebook/opt-1.3b \
        --platform=shark
```
Any OPT model from huggingface should work with this script, and you can choose between `--platform=shark` or `--platform=huggingface` to generate benchmarks of OPT inference on SHARK / PyTorch. 

# Run a small suite of OPT models through the benchmark script

```
python opt_perf_comparison_batch.py
```
This script will run benchmarks from a suite of OPT configurations:
- Sequence Lengths: 32, 128, 256, 512
- Parameter Counts: 125m, 350m, 1.3b

note: Most of these scripts are written for use on CPU, as perf comparisons against pytorch can be problematic across platforms otherwise.
