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



####1111 Merge all files together and add the region column
#print("11111111111111")
folder_path = 'archive/'  # Replace with the actual folder path
desired_columns = ['title', 'category_id', 'description','tags','views','likes','dislikes','comment_count']  # Replace with the desired column names

csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

new_data_frame = pd.DataFrame()
#print("aaaaaaaaaaaaaaa")
#i=1
for file in csv_files:
    #print(i)
    #i=i+1
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df['region'] = file  # Add a new column named 'region' and use the file name as its content
    desired_data = df[desired_columns + ['region']]  # Include the desired columns and the 'region' column
    new_data_frame = pd.concat([new_data_frame, desired_data], ignore_index=True)

#new_file_name = 'new_file.csv'  # Replace with the desired name for the new CSV file
#new_data_frame.to_csv(new_file_name, index=False)
#print("333333333")




####2222 Replace the file names in the 'region' column with the actual region names
# Create a dictionary to map the new replacement values
replacement_dict = {
    'CAvideos.csv': 'Canada',
    'DEvideos.csv': 'Germany',
    'FRvideos.csv': 'France',
    'GBvideos.csv': 'Great Britain',
    'INvideos.csv': 'India',
    'MXvideos.csv': 'Mexico',
    'KRvideos.csv': 'South Korea',
    'JPvideos.csv': 'Japan',
    'RUvideos.csv': 'Russia',
    'USvideos.csv': 'USA'
}


# Use the dictionary to replace values in the 'region' column
new_data_frame['region'] = new_data_frame['region'].map(replacement_dict)

# Write the modified DataFrame to a new CSV file
#new_data_frame.to_csv('new_file_modified.csv', index=False)




####3333 Handle non-string parts in the data
df = new_data_frame
# Define the columns to be processed
columns_to_convert = ['title', 'description']
#columns_to_convert = ['title']

# Convert non-string values in the specified columns to NaN
df[columns_to_convert] = df[columns_to_convert].astype(str)

# Drop rows containing NaN values
df = df.dropna(subset=columns_to_convert)
df.to_csv('new_file_modified.csv', index=False)



#df = pd.read_csv('new_file_modified.csv')




####4444 Sentiment analysis
# Define the sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(str(text))  
    sentiment = blob.sentiment.polarity
    return sentiment

# Perform sentiment analysis on the titles and create a new column
df.loc[:, 'title_sentiment'] = df['title'].apply(analyze_sentiment)


# Perform sentiment analysis on the descriptions and create a new column
df.loc[:, 'description_sentiment'] = df['description'].apply(analyze_sentiment)

# Write the modified DataFrame to a new CSV file
df.to_csv('after_sentiment_analysis.csv', index=False)