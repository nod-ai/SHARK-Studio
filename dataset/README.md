# Dataset annotation tool

SHARK annotator for adding or modifying prompts of dataset images

## Set up

Activate SHARK Python virtual environment and install additional packages
```shell
source ../shark.venv/bin/activate
pip install -r requirements.txt
```

## Run annotator

```shell
python annotation_tool.py
```

<img width="1280" alt="annotator" src="https://user-images.githubusercontent.com/49575973/214521137-7ef6ae10-7cd8-46e6-b270-b6c0445157f1.png">

* Select a dataset from `Dataset` dropdown list
* Select an image from `Image` dropdown list
* Image and the existing prompt will be loaded
* Select a prompt from `Prompt` dropdown list to modify or "Add new" to add a prompt
* Click `Save` to save changes, click `Delete` to delete prompt
* Click `Back` or `Next` to switch image, you could also select other images from `Image`
* Click `Finish` when finishing annotation or before switching dataset
