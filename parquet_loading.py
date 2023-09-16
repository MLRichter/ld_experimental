import pandas as pd

# Read the Parquet file into a pandas DataFrame
#df = pd.read_parquet("../coco2017/coco_30k.parquet", engine="pyarrow")
df = pd.read_parquet("../coco2017/long_context_val.parquet", engine="pyarrow")
captions = df['caption'].values
# Show the DataFrame
print(df)
#print(captions)
print(df[df["file_name"] == "COCO_val2014_000000007961.jpg"]["caption"].values[0])


import torch

