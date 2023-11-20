import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Sample data for the two CSV files
data_csv1 = "./output/clip_spl_wuerstchen.csv"  # The first CSV data
data_csv2 = "./output/fid_spl.csv"  # The second CSV data

# Convert the string data to pandas DataFrames
df1 = pd.read_csv(data_csv1, sep=';', index_col=0)
df2 = pd.read_csv(data_csv2, sep=';', index_col=0)

# Extract the first word from the model name for grouping
df1['group'] = df1['model'].apply(lambda x: x.split('_')[0])
df2['group'] = df2['model'].apply(lambda x: x.split('_')[0])

# Merge the two DataFrames based on the model name
merged_df = pd.merge(df1, df2, on='model')

# Group by the first word of the model name
grouped = merged_df.groupby('group_x')

# Create a single figure for both groups
plt.figure(figsize=(7, 5))

print(len(grouped))
mapper = {"wuerstchen": "Würstchen", "sd21": "Stable Diffusion 2.1"}


def y_offset(annot: str) -> float:
    if "80" in annot:
        return 1
    elif "spl=160" in annot:
        return 1
    elif "spl=40" in annot:
        return -1
    elif "spl=20" in annot:
        return -2.6
    elif "spl=10" in annot:
        return -2.6
    else:
        return 0

def x_offset(annot: str) -> float:
    if "160" in annot:
        return -0.005
    #if "40" in annot:
    #    return -0.005
    elif "spl=20" in annot:
        return -0.001
    if "10" in annot:
        return -0.002
    else:
        return 0.001

# Iterate over each group to plot their data
for name, group in grouped:
    plt.scatter(group['clip-score'], group['fid'], label=mapper[name], color='orange')

    # Annotate each point
    for i, row in group.iterrows():
        annotation = f"spl={row['model'].split('_')[1].replace('spl', '')}"
        plt.annotate(annotation, (row['clip-score']+x_offset(annotation),
                                  row['fid']+y_offset(annotation)), fontsize=12)

    # Draw lines between consecutive points
    sorted_group = group#.sort_values(by=['model'])
    plt.plot(sorted_group['clip-score'], sorted_group['fid'], marker='',
             color='orange',
             #linewidth=8,
             #alpha=0.5
             )

# Set title, labels, and legend
#plt.xlim((0.215, 0.273))
#plt.yscale("log")
plt.xlabel('Clip Score (ViT-G/14)', fontsize=16)
plt.ylabel('FID Score', fontsize=16)
plt.legend(fontsize=16)

# Show the plot
plt.tight_layout()
plt.savefig("spl_fid.pdf")
plt.show()
