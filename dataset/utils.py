from google.cloud import storage


def get_datasets(gs_url):
    datasets = set()
    images = dict()
    ds_w_prompts = []

    storage_client = storage.Client()
    bucket_name = gs_url.split("/")[2]
    source_blob_name = "/".join(gs_url.split("/")[3:])
    blobs = storage_client.list_blobs(bucket_name, prefix=source_blob_name)

    for blob in blobs:
        dataset_name = blob.name.split("/")[1]
        if dataset_name == "":
            continue
        datasets.add(dataset_name)
        if dataset_name not in images.keys():
            images[dataset_name] = []

        # check if image or jsonl
        file_sub_path = "/".join(blob.name.split("/")[2:])
        if "/" in file_sub_path:
            images[dataset_name] += [file_sub_path]
        elif "metadata.jsonl" in file_sub_path:
            ds_w_prompts.append(dataset_name)

    return list(datasets), images, ds_w_prompts
