In order to launch SHARK-web, from the root SHARK directory, run:

```shell
IMPORTER=1 ./setup_venv.sh
source shark.venv/bin/activate
cd web
python index.py
```

This will launch a gradio server with a public URL like:
Running on public URL: https://xxxxx.gradio.app
