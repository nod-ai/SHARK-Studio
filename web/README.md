In order to launch SHARK-web, from the root SHARK directory, run:

## Linux
```shell
IMPORTER=1 ./setup_venv.sh
source shark.venv/bin/activate
pip install opencv-python gdown
cd web
gdown 1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene
python index.py
```

## Windows
```shell
./setup_venv.ps1
cd web
python index.py --local_tank_cache=<current_working_dir>
```
