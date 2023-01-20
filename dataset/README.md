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
* Select dataset from `Dataset` dropdown list
* Select image from `Image` dropdown list
* Image and the existing prompt will be loaded
* Add or modify prompt in `Prompt` textbox which will be autosaved
* Click `Next` to load the next image, you could also select other images from `Image`
* Click `Finish` when finishing annotation or before switching dataset
