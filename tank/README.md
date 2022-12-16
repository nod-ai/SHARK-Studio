## Supported and Validated Models

### PyTorch HuggingFace Models

| PyTorch Language Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :green_heart: (JIT)          | :green_heart:         | :green_heart:         | :green_heart:            |
| Albert              | :green_heart: (JIT)            | :green_heart:         | :green_heart:         | :green_heart:            |
| BigBird             | :green_heart: (AOT)            |          |          |             |
| dbmdz/ConvBERT      | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| DistilBERT          | :broken_heart: (JIT)            |          |          |             |
| GPT2                | :green_heart:            | :green_heart:         |  :green_heart:        | :green_heart:            |
| MobileBert          | :green_heart: (JIT)            | :green_heart:         | :green_heart:         | :green_heart:            |
| microsoft/beit      | :green_heart:                  | :green_heart:         | :broken_heart:         | :broken_heart:            |
| facebook/deit       | :green_heart:          | :green_heart:         | :broken_heart:         | :broken_heart:            |
| facebook/convnext   | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |

### Torchvision  Models

| TORCHVISION Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|--------------------|----------------------|----------|----------|-------------|
| AlexNet            | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| MobileNetV2        | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| MobileNetV3        | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| Unet               | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| Resnet18           | :green_heart: (Script)         | :green_heart:         |  :green_heart:        | :green_heart:            |
| Resnet50           | :green_heart: (Script)         | :green_heart:         |   :green_heart:       | :green_heart:            |
| Resnet101           | :green_heart: (Script)         | :green_heart:         |   :green_heart:       | :green_heart:            |
| Resnext50_32x4d    | :green_heart: (Script)         |          |          |             |
| SqueezeNet         | :green_heart: (Script)         | :green_heart:         |   :broken_heart:       | :broken_heart:            |
| EfficientNet       | :green_heart: (Script)         |          |          |             |
| Regnet             | :green_heart: (Script)         |          |          |             |
| Resnest            | :broken_heart: (Script)         |          |          |             |
| Vision Transformer | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| VGG 16             | :green_heart: (Script)         | :green_heart:         |   :green_heart:       |             |
| Wide Resnet        | :green_heart: (Script)         | :green_heart:         | :green_heart:         | :green_heart:            |
| RAFT               | :broken_heart: (JIT)            |          |          |             |

For more information refer to [MODEL TRACKING SHEET](https://docs.google.com/spreadsheets/d/15PcjKeHZIrB5LfDyuw7DGEEE8XnQEX2aX8lm8qbxV8A/edit#gid=0)

### Tensorflow Models (Inference)

| Hugging Face Models | tf-mhlo lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| MiniLM                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| albert-base-v2              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| DistilBERT          | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| CamemBert                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| ConvBert              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| Deberta              |            |         |          |             |
| electra          | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| funnel              |            |         |          |             |
| layoutlm              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| longformer              |            |         |          |             |
| mobile-bert                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| rembert              |            |         |          |             |
| tapas              |            |         |          |             |
| flaubert                | :broken_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| roberta                | :green_heart:          | :green_heart:         | :green_heart:         | :green_heart:            |
| xlm-roberta              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |
| mpnet              | :green_heart:            | :green_heart:         | :green_heart:         | :green_heart:            |

### PyTorch Training Models

| Models | Torch-MLIR lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :green_heart:           | :green_heart:         |          |             |
| FullyConnected                | :green_heart:           | :green_heart:         |          |             |

### JAX  Models

| Models | JAX-MHLO lowerable | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| DALL-E                | :broken_heart:           | :broken_heart:         |          |             |
| FullyConnected                | :green_heart:           | :green_heart:         |          |             |

<details>
  <summary>TFLite Models</summary>

### TFLite Models

| Models | TOSA/LinAlg  | SHARK-CPU | SHARK-CUDA | SHARK-METAL |
|---------------------|----------------------|----------|----------|-------------|
| BERT                | :broken_heart:           | :broken_heart:         |          |             |
| FullyConnected      | :green_heart:           | :green_heart:         |          |             |
| albert | :green_heart:           | :green_heart:         |          |             |
| asr_conformer | :green_heart:           | :green_heart:         |          |             |
| bird_classifier | :green_heart:           | :green_heart:         |          |             |
| cartoon_gan | :green_heart:           | :green_heart:         |          |             |
| craft_text | :green_heart:           | :green_heart:         |          |             |
| deeplab_v3 | :green_heart:           | :green_heart:         |          |             |
| densenet | :green_heart:           | :green_heart:         |          |             |
| east_text_detector | :green_heart:           | :green_heart:         |          |             |
| efficientnet_lite0_int8 | :green_heart:           | :green_heart:         |          |             |
| efficientnet | :green_heart:           | :green_heart:         |          |             |
| gpt2 | :green_heart:           | :green_heart:         |          |             |
| image_stylization | :green_heart:           | :green_heart:         |          |             |
| inception_v4 | :green_heart:           | :green_heart:         |          |             |
| inception_v4_uint8 | :green_heart:           | :green_heart:         |          |             |
| lightning_fp16 | :green_heart:           | :green_heart:         |          |             |
| lightning_i8 | :green_heart:           | :green_heart:         |          |             |
| lightning | :green_heart:           | :green_heart:         |          |             |
| magenta | :green_heart:           | :green_heart:         |          |             |
| midas | :green_heart:           | :green_heart:         |          |             |
| mirnet | :green_heart:           | :green_heart:         |          |             |
| mnasnet | :green_heart:           | :green_heart:         |          |             |
| mobilebert_edgetpu_s_float | :green_heart:           | :green_heart:         |          |             |
| mobilebert_edgetpu_s_quant | :green_heart:           | :green_heart:         |          |             |
| mobilebert | :green_heart:           | :green_heart:         |          |             |
| mobilebert_tf2_float | :green_heart:           | :green_heart:         |          |             |
| mobilebert_tf2_quant | :green_heart:           | :green_heart:         |          |             |
| mobilenet_ssd_quant | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v1 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v1_uint8 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v2_int8 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v2 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v2_uint8 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v3-large | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v3-large_uint8 | :green_heart:           | :green_heart:         |          |             |
| mobilenet_v35-int8 | :green_heart:           | :green_heart:         |          |             |
| nasnet | :green_heart:           | :green_heart:         |          |             |
| person_detect | :green_heart:           | :green_heart:         |          |             |
| posenet | :green_heart:           | :green_heart:         |          |             |
| resnet_50_int8 | :green_heart:           | :green_heart:         |          |             |
| rosetta | :green_heart:           | :green_heart:         |          |             |
| spice | :green_heart:           | :green_heart:         |          |             |
| squeezenet | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v1 | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v1_uint8 | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v2_fpnlite | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v2_fpnlite_uint8 | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v2_int8 | :green_heart:           | :green_heart:         |          |             |
| ssd_mobilenet_v2 | :green_heart:           | :green_heart:         |          |             |
| ssd_spaghettinet_large | :green_heart:           | :green_heart:         |          |             |
| ssd_spaghettinet_large_uint8 | :green_heart:           | :green_heart:         |          |             |
| visual_wake_words_i8 | :green_heart:           | :green_heart:         |          |             |

</details>

## Testing and Benchmarks

### Run all model tests on CPU/GPU/VULKAN/Metal

For a list of models included in our pytest model suite, see https://github.com/nod-ai/SHARK/blob/main/tank/all_models.csv

```shell
pytest tank/test_models.py

# Models included in the pytest suite can be found listed in all_models.csv.

# If on Linux for multithreading on CPU (faster results):
pytest tank/test_models.py -n auto
```

### Running specific tests
```shell

# Search for test cases by including a keyword that matches all or part of the test case's name;
pytest tank/test_models.py -k "keyword" 

# Test cases are named uniformly by format test_module_<model_name_underscores_only>_<torch/tf>_<static/dynamic>_<device>.

# Example: Test all models on nvidia gpu:
pytest tank/test_models.py -k "cuda"

# Example: Test all tensorflow resnet models on Vulkan backend:
pytest tank/test_models.py -k "resnet and tf and vulkan"

# Exclude a test case:
pytest tank/test_models.py -k "not ..."

### Run benchmarks on SHARK tank pytests and generate bench_results.csv with results.

(the following requires source installation with `IMPORTER=1 ./setup_venv.sh`)

```shell
pytest --benchmark tank/test_models.py
  
# Just do static GPU benchmarks for PyTorch tests:
pytest --benchmark tank/test_models.py -k "pytorch and static and cuda"

```
  
### Benchmark Resnet50, MiniLM on CPU

(requires source installation with `IMPORTER=1 ./setup_venv.sh`)  
  
```shell
# We suggest running the following commands as root before running benchmarks on CPU:
  
cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list | awk -F, '{print $2}' | sort -n | uniq | ( while read X ; do echo $X ; echo 0 > /sys/devices/system/cpu/cpu$X/online ; done )
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo

# Benchmark canonical Resnet50 on CPU via pytest
pytest --benchmark tank/test_models.py -k "resnet50 and tf_static_cpu"

# Benchmark canonical MiniLM on CPU via pytest
pytest --benchmark tank/test_models.py -k "MiniLM and cpu"

# Benchmark MiniLM on CPU via transformer-benchmarks:
git clone --recursive https://github.com/nod-ai/transformer-benchmarks.git
cd transformer-benchmarks
./perf-ci.sh -n
# Check detail.csv for MLIR/IREE results.

```

To run the fine tuning example, from the root SHARK directory, run:

```shell
IMPORTER=1 ./setup_venv.sh
source shark.venv/bin/activate
pip install jupyter tf-models-nightly tf-datasets
jupyter-notebook
```
if running from a google vm, you can view jupyter notebooks on your local system with:
```shell
gcloud compute ssh <YOUR_INSTANCE_DETAILS> --ssh-flag="-N -L localhost:8888:localhost:8888"
```



