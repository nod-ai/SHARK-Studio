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

