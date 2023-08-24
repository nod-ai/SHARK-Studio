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
