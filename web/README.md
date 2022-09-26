In order to launch SHARK-web, from the root SHARK directory, run:

```shell
IMPORTER=1 ./setup_venv.sh
source shark.venv/bin/activate
pip install diffusers scipy
cd web
wget -O models_mlir/stable_diffusion.mlir https://storage.googleapis.com/shark_tank/prashant_nod/stable_diff/stable_diff_torch.mlir
python index.py
```
This will launch a gradio server with a public URL.
