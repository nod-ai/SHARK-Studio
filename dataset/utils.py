from google.cloud import storage


def get_datasets(gs_url):
    datasets = set()
    images = dict()

    storage_client = storage.Client()
    bucket_name = gs_url.split("/")[2]
    source_blob_name = "/".join(gs_url.split("/")[3:])
    blobs = storage_client.list_blobs(bucket_name, prefix=source_blob_name)

    for blob in blobs:
        dataset_name = blob.name.split("/")[1]
        datasets.add(dataset_name)
        file_sub_path = "/".join(blob.name.split("/")[2:])
        # check if image or jsonl
        if "/" in file_sub_path:
            if dataset_name not in images.keys():
                images[dataset_name] = []
            images[dataset_name] += [file_sub_path]

    return list(datasets), images
