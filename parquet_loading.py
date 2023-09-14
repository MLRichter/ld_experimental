import pandas as pd

# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet("../coco2017/coco_30k.parquet", engine="pyarrow")
captions = df['caption'].values
# Show the DataFrame
print(df)
#print(captions)
print(df[df["file_name"] == "COCO_val2014_000000001398.jpg"])


import torch

