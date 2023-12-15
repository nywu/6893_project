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


####7777 Group by the 'cluster' column and calculate popularity metrics for each label (e.g., average views, average likes, etc.)
selected_region = 'South Korea'
#selected_region_rows_df = pd.read_csv(f'{selected_region}_tags_modified.csv', encoding='latin1')

selected_region_rows_df = pd.read_csv(f'{selected_region}_tags_modified.csv')

grouped = selected_region_rows_df.groupby('cluster').agg({
    'views': 'mean',
    'likes': 'mean',
    'dislikes': 'mean',
    'comment_count': 'mean'
}).reset_index()

grouped.to_csv(f'{selected_region}_means.csv', index=False)




####8888 Get the published_time for the selected region and preprocess it

#selected_region = 'Canada'
#selected_region_rows_df=pd.read_csv(f'{selected_region}_tags_modified.csv')

folder_path = 'archive/'  # Replace with the actual folder path
reversed_dict = {
    'Canada': 'CAvideos.csv',
    'Germany': 'DEvideos.csv',
    'France': 'FRvideos.csv',
    'Great Britain': 'GBvideos.csv',
    'India': 'INvideos.csv',
    'Mexico': 'MXvideos.csv',
    'South Korea': 'KRvideos.csv',
    'Japan': 'JPvideos.csv',
    'Russia': 'RUvideos.csv',
    'USA': 'USvideos.csv'
}
file_name = reversed_dict[selected_region]
file_path = os.path.join(folder_path, file_name)
region_df = pd.read_csv(file_path)

region_df['trending_date'] = pd.to_datetime(region_df['trending_date'], format='%y.%d.%m')
region_df['publish_time'] = pd.to_datetime(region_df['publish_time']).dt.tz_localize(None)

# Set 'trending_date' to 23:59:59 for all rows
region_df['trending_date'] = region_df['trending_date'].dt.floor('D') + pd.Timedelta(hours=23, minutes=59, seconds=59)
region_df['time_difference'] = region_df['trending_date'] - region_df['publish_time']
#print(region_df['time_difference'])
#print(region_df['trending_date'])
#print(region_df['publish_time'])

selected_region_rows_df['total_published_time'] = region_df['time_difference'].dt.total_seconds() + region_df['time_difference'].dt.days * 24 * 60 * 60

# scaler
scaler = StandardScaler()
selected_region_rows_df.loc[:, 'normalized_total_published_time'] = scaler.fit_transform(selected_region_rows_df[['total_published_time']])





####9999 Take the logarithm of views, dislikes, and comment_count for the selected region

#selected_region = 'Canada'
#selected_region_rows_df=pd.read_csv(f'{selected_region}_tags_modified.csv')

# Logarithm
selected_region_rows_df['views'] = np.log1p(selected_region_rows_df['views'])
selected_region_rows_df['dislikes'] = np.log1p(selected_region_rows_df['dislikes'])
selected_region_rows_df['comment_count'] = np.log1p(selected_region_rows_df['comment_count'])

# Write the modified DataFrame to a new CSV file
selected_region_rows_df.to_csv(f'{selected_region}_tags_modified.csv', index=False)