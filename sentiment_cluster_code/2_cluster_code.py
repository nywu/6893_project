import pandas as pd
import os
import json
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from googletrans import Translator
from scipy.sparse import csr_matrix

# 5 f'{selected_region}_rows.csv' Data for a specific region (including sentiment)
# 6 f'{selected_region}_tags_modified.csv' Data for a specific region (including sentiment and cluster)

df = pd.read_csv('after_sentiment_analysis.csv') 

####5555 Select data for a specific region
# Select all rows for a specific region (e.g., region is 'Canada')
selected_region = 'Canada'
selected_region_rows = df[df['region'] == selected_region]

# Save the selected rows in "selected_region_rows"
selected_region_rows_df = selected_region_rows.copy()
selected_region_rows_df['tags'] = selected_region_rows_df['tags'].astype(str)

# Remove rows with empty tags or "NONE"
selected_region_rows_df = selected_region_rows_df[(selected_region_rows_df['tags'] != '') & (selected_region_rows_df['tags'] != '[none]')]

# Save "selected_region_rows" as a CSV file
selected_region_rows_df.to_csv(f'{selected_region}_rows.csv', index=False)



#selected_region = 'Canada'
#selected_region_rows_df = pd.read_csv(f'{selected_region}_rows.csv') 
####6666 Cluster tags for the selected region into 20 clusters
# Extract the 'tags' column as text data
tags = selected_region_rows_df['tags']
#print(tags)
#print(type(tags))

# Split the string into multiple tags using split() method
tags_list = tags.apply(lambda x: x.split('|'))

# Remove quotes and spaces from the tags
tags_list = tags_list.apply(lambda x: [tag.strip('\" ') for tag in x])
#print(tags_list)
#print(type(tags_list))

# Keep only ten elements for each row, repeat if there are fewer than ten
tags_list = tags_list.apply(lambda x: (x * (10 // len(x) + 1))[:10])
#print(tags_list)


# Create a TF-IDF feature extractor with the same settings
vectorizer = TfidfVectorizer()

# Create an empty 2D array
result_array = []

# Iterate over each list in tags_list
for tags in tags_list:
    # Extract text features
    features = vectorizer.fit_transform(tags)

    # Extract nonzero elements of the sparse matrix
    nonzero_indices = features.nonzero()
    nonzero_values = features[nonzero_indices]

    # Convert nonzero values to a NumPy array and resize
    features_array = np.array(nonzero_values)
    features_array = np.resize(features_array, 10)

    # Add features_array to the result array
    result_array.append(features_array)

# Convert the result array to a NumPy 2D array
result_array = np.array(result_array)

# Output the result array
#print(result_array)
#print(type(result_array))


# Create a K-Means clustering model with the specified number of clusters
kmeans = KMeans(n_clusters=20)

# Cluster the features
kmeans.fit(result_array)

# Get the cluster labels for each sample
cluster_labels = kmeans.labels_

# Save the clustering model to a file
filename = f'{selected_region}_kmeans_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(kmeans, file)

# Add the cluster results to the original DataFrame
selected_region_rows_df['cluster'] = kmeans.labels_

# Write the modified DataFrame to a new CSV file
selected_region_rows_df.to_csv(f'{selected_region}_tags_modified.csv', index=False)