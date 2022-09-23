1. Install torchdynamo
   - `git clone https://github.com/pytorch/torchdynamo.git`
   - `cd torchdynamo`
   - `python -m pip install -r requirements.txt`
   - `python setup.py develop`

2. Install functorch
   - `python -m pip install -v "git+https://github.com/pytorch/pytorch.git@$(python -c "import torch.version; print(torch.version.git_version)")#subdirectory=functorch"`

3. Run examples.
    - `python shark/examples/shark_dynamo/basic_examples.py`
