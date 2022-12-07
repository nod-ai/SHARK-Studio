from google.cloud import storage
from glob import glob
import os


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


targets_file_dir = os.path.join(os.getcwd(), "temp/repository/targets")
targets_files = glob(os.path.join(targets_file_dir, "*"))

metadata_file_dir = os.path.join(os.getcwd(), "temp/repository/metadata")
metadata_files = glob(os.path.join(metadata_file_dir, "*"))


for file in metadata_files:
    filename = os.path.split(file)[-1]
    destination_blob_name = "/releases/metadata/" + filename
    upload_blob("shark_tank", file, destination_blob_name)

for file in targets_files:
    filename = os.path.split(file)[-1]
    destination_blob_name = "/releases/targets/" + filename
    upload_blob("shark_tank", file, destination_blob_name)
