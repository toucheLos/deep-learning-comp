import pandas as pd
import json

# Load the JSON data
with open('~/dl-comp/results/resnet_50_results.json') as f:
    data = json.load(f)

# Convert JSON data to pandas DataFrame
df = pd.json_normalize(data['Configurations'])

# Analyze the DataFrame
print(df.describe())
