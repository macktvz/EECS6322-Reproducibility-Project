from pathlib import Path
from datasets import load_dataset

from reproducibility_project.config import DATA_PATH

# this should be more than 1000 images
file_names = [
    ".part-00000-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet.crc",
    ".part-00021-339dc23d-0869-4dc0-9390-b4036fcc80c2-c000.snappy.parquet.crc"
]
# total size 500gb, for 20m imgs so each img is 
DATASET_SIZE = 1000
ds = load_dataset("laion/relaion2B-en-research-safe", data_files=file_names)
assert(len(ds) > DATASET_SIZE)