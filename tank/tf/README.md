## Running SharkInference on CPUs, GPUs and MAC.


### Run the binary sequence_classification.
#### The models supported are: [hugging face sequence classification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.TFAutoModelForSequenceClassification)
```shell
./seq_classification.py --hf_model_name="hf_model" --device="cpu" # Use gpu | vulkan
```

Once the model is compiled to run on the device mentioned, we can pass in text and 
get the logits.




