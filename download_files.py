import os
import argparse
from google.cloud import storage

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

parser = argparse.ArgumentParser()
parser.add_argument('--model_size', default="11B")
parser.add_argument('--model_step', default="1100500")

args = parser.parse_args()

bucket_name = "unifiedqa"
size = args.model_size
step = args.model_step
source_dir = f"models/{size}/"
destination_dir = f"models/unifiedqa_trained/{size}/"
file_name_list = [f"model.ckpt-{step}.data-00000-of-00002", f"model.ckpt-{step}.data-00001-of-00002", f"model.ckpt-{step}.meta", f"model.ckpt-{step}.index"]

if not os.path.isdir(destination_dir):
    os.mkdir(destination_dir)

for file_name in file_name_list:
    source_blob_name = source_dir+file_name
    destination_file_name = destination_dir+file_name
    print("Downloading", bucket_name,source_blob_name)
    download_blob(bucket_name, source_blob_name, destination_file_name)
'''
model.ckpt-1100500.meta
model.ckpt-1100500.index
model.ckpt-1100500.data-00001-of-00002
model.ckpt-1100500.data-00002-of-00002
'''