import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Sample data for the two CSV files
data_csv1 = "./output/clip_cfg_wuerstchen_scores2.csv"  # The first CSV data
data_csv2 = "./output/fid_cfg.csv"  # The second CSV data

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

# Iterate over each group to plot their data
for name, group in grouped:
    plt.scatter(group['clip-score'], group['fid'], label=mapper[name])

    # Annotate each point
    for i, row in group.iterrows():
        annotation = f"cfg={row['model'].split('_')[1]}"
        plt.annotate(annotation, (row['clip-score']+0.001, row['fid']), fontsize=12)

    # Draw lines between consecutive points
    sorted_group = group.sort_values(by=['clip-score'])
    plt.plot(sorted_group['clip-score'], sorted_group['fid'], marker='',
             #color='orange',
             #linewidth=8,
             #alpha=0.5
             )

# Set title, labels, and legend
plt.xlim((0.215, 0.273))
plt.xlabel('Clip Score (ViT-G/14)', fontsize=16)
plt.ylabel('FID Score', fontsize=16)
plt.legend(fontsize=16)


# Show the plot
plt.tight_layout()
plt.savefig("cfg_fid.pdf")
plt.show()
