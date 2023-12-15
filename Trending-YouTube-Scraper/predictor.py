import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def perform_predict(data_dir, file_dir, model_dir, public_duration):
    # List CSV files in the specified directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith(('US_', 'CA_', 'FR_', 'IN_'))]
    # Initialize lists to store HTML content
    html_content_list = []

    # Iterate through each CSV file
    for csv_file in csv_files:
        # Construct the full path to the CSV file
        data_path = os.path.join(data_dir, csv_file)

        # Load data from CSV file
        videos = pd.read_csv(data_path, encoding='utf-8')
        selected_region = csv_file[:2]

        # Find the index of the maximum 'likes' for each group
        max_likes_index = videos.groupby(['title', 'thumbnail_link'])['likes'].idxmax()
        # Extract the rows with the maximum 'likes' for each group
        mvideo = videos.loc[max_likes_index].reset_index(drop=True)

        mvideo['Total_Likes'] = round(mvideo['likes'], 2)
        mvideo = mvideo.sort_values(by='Total_Likes', ascending=False)

        # Select top 10 videos
        top_10_videos = mvideo.head(10).reset_index(drop=True)

        for index, row in top_10_videos.iterrows():
            top_10_videos.loc[index, 'cluster'] = cluster_tags(selected_region, row, model_dir)

        # prediction
        processed_data = process_data(top_10_videos, public_duration)
        top_10_videos['Future_Likes'] = (make_predictions(processed_data, selected_region, model_dir)).astype(int)
        print(top_10_videos)

        # Create a new column 'image' with HTML image tags
        top_10_videos['image'] = '<img width="80%" height="80%" src="' + top_10_videos['thumbnail_link'] + '"></img>'

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Top 10 YouTube Trending Videos in {selected_region}</title>
        </head>
        <body>
            <h2>Top 10 YouTube Trending Videos in {selected_region}</h2>
            <table class="display nowrap hover row-border">
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Title</th>
                  <th>Total Likes</th>
                  <th>Future Likes (After {public_duration} days)</th>
                </tr>
              </thead>
              <tbody>
        """

        for _, row in top_10_videos.iterrows():
            html_content += f"<tr><td>{row['image']}</td><td>{row['title']}</td><td>{row['Total_Likes']}</td><td>{row['Future_Likes']}</td></tr>"

        html_content += """
              </tbody>
            </table>
        </body>
        </html>
        """

        # Append the HTML content to the list
        html_content_list.append(html_content)

    # Combine all HTML content into a single string
    combined_html_content = "\n".join(html_content_list)

    # Save the combined HTML content to a file in the output directory
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    combined_html_path = os.path.join(file_dir, 'top10_predict.html')
    with open(combined_html_path, 'w') as f:
        f.write(combined_html_content)


def perform_eda(data_dir, file_dir):
    # List CSV files in the specified directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith(('US_', 'CA_', 'FR_', 'IN_'))]
    # Initialize lists to store plot paths and HTML content
    html_content_list = []

    # Iterate through each CSV file
    for csv_file in csv_files:
        # Construct the full path to the CSV file
        data_path = os.path.join(data_dir, csv_file)

        # Load data from CSV file
        videos = pd.read_csv(data_path, encoding='utf-8')
        selected_region = csv_file[:2]

        # Find the index of the maximum 'likes' for each group
        max_likes_index = videos.groupby(['title', 'thumbnail_link'])['likes'].idxmax()
        # Extract the rows with the maximum 'likes' for each group
        mvideo = videos.loc[max_likes_index].reset_index(drop=True)

        mvideo['Total_Likes'] = round(mvideo['likes'], 2)
        mvideo = mvideo.sort_values(by='Total_Likes', ascending=False)

        # Select top 10 videos
        top_10_videos = mvideo.head(10).reset_index(drop=True)
        print(top_10_videos)

        # Create a new column 'image' with HTML image tags
        top_10_videos['image'] = '<img width="80%" height="80%" src="' + top_10_videos['thumbnail_link'] + '"></img>'

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Top 10 YouTube Trending Videos in {selected_region}</title>
        </head>
        <body>
            <h2>Top 10 YouTube Trending Videos in {selected_region}</h2>
            <table class="display nowrap hover row-border">
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Title</th>
                  <th>Total Likes</th>
                </tr>
              </thead>
              <tbody>
        """

        for _, row in top_10_videos.iterrows():
            html_content += f"<tr><td>{row['image']}</td><td>{row['title']}</td><td>{row['Total_Likes']}</td></tr>"

        html_content += """
              </tbody>
            </table>
        </body>
        </html>
        """

        # Append the HTML content to the list
        html_content_list.append(html_content)

    # Combine all HTML content into a single string
    combined_html_content = "\n".join(html_content_list)

    # Save the combined HTML content to a file in the output directory
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    combined_html_path = os.path.join(file_dir, 'top10_liked.html')
    with open(combined_html_path, 'w') as f:
        f.write(combined_html_content)


def cluster_tags(selected_region, raw_data, model_dir):
    tags = raw_data['tags']
    # get tag cluster
    tags_list = tags.split('|')
    tags_list = [tag.strip('\" ') for tag in tags_list]

    # get 10 tags
    if len(tags_list) < 10:
        tags_list = tags_list * (10 // len(tags_list)) + tags_list[:10 % len(tags_list)]
    else:
        tags_list = tags_list[:10]

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(tags_list)

    nonzero_indices = features.nonzero()
    nonzero_values = features[nonzero_indices]

    features_array = np.array(nonzero_values)
    features_array = np.resize(features_array, 10)

    # load models
    filename = os.path.join(model_dir, f'{selected_region}_kmeans_model.pkl')
    with open(filename, 'rb') as file:
        kmeans = pickle.load(file)
    # get cluster
    features_array = features_array.reshape(1, -1)
    cluster = kmeans.predict(features_array)[0]

    return cluster


def process_data(raw_data, public_duration):
    views = raw_data['view_count']
    dislikes = raw_data['dislikes']
    comment_count = raw_data['comment_count']
    title = raw_data['title']
    description = raw_data['description']
    trending_date = raw_data['trending_date']
    publish_time = raw_data['publishedAt']
    categoryId = raw_data['categoryId']
    cluster = raw_data['cluster']

    # log
    views = np.log1p(views)
    dislikes = np.log1p(dislikes)
    comment_count = np.log1p(comment_count)

    # sentiment
    def analyze_sentiment(text):
        blob = TextBlob(str(text))
        sentiment = blob.sentiment.polarity
        return sentiment

    title_sentiment = title.apply(analyze_sentiment)
    description_sentiment = description.apply(analyze_sentiment)

    # published time
    trending_date = trending_date.apply(lambda x: datetime.strptime(x, '%y.%d.%m'))
    trending_date = trending_date.apply(lambda x: x.replace(hour=23, minute=59, second=59))
    publish_time = publish_time.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
    time_difference = trending_date - publish_time

    # Calculate the total seconds element-wise
    normalized_total_published_time = time_difference.dt.total_seconds()
    normalized_total_published_time = normalized_total_published_time + pd.Timedelta(days=public_duration).total_seconds()

    # Standardize the time difference
    scaler = StandardScaler()
    normalized_total_published_time = scaler.fit_transform(normalized_total_published_time.values.reshape(-1, 1))
    normalized_total_published_time = pd.Series(normalized_total_published_time[:, 0])

    # return a preprocessed dataframe with tag cluster
    datadf = pd.DataFrame({
        'category_id': categoryId,
        'views': views,
        'normalized_total_published_time': normalized_total_published_time,
        'title_sentiment': title_sentiment,
        'description_sentiment': description_sentiment,
        'comment_count': comment_count,
        'dislikes': dislikes,
        'cluster': cluster
    })
    return datadf


def make_predictions(df, selected_region, model_dir):
    # one-hot category
    df['category_id'] = df['category_id'].astype(str)
    encoded_data = pd.get_dummies(df, columns=['category_id'], prefix='category')

    # load json
    file_name = os.path.join(model_dir, selected_region + '_category_id.json')

    if file_name is not None:
        with open(file_name, 'r') as f:
            data = json.load(f)
            max_id = max(int(category['id']) for category in data['items'])
    else:
        print(f"No JSON file found for region: {selected_region}")

    all_categories = range(1, max_id + 1)
    for category in all_categories:
        column_name = f'category_{category}'
        if column_name not in encoded_data.columns:
            encoded_data[column_name] = 0

    # tag
    encoded_data['cluster'] = encoded_data['cluster'].astype(str)
    data_to_predict = pd.get_dummies(encoded_data, columns=['cluster'], prefix='cluster')

    for cluster in range(0, 20):
        column_name = f'cluster_{cluster}'
        if column_name not in data_to_predict.columns:
            data_to_predict[column_name] = 0

    model_path = os.path.join(model_dir, f'{selected_region}model.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    feature_order = model.feature_names_in_
    data_to_predict = data_to_predict[feature_order]
    predictions = model.predict(data_to_predict)
    likes = np.round(np.expm1(predictions)*10)

    return likes


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--key_path',
#                         help='Path to the file containing the api key, by default will use api_key.txt in the same directory',
#                         default='api_key.txt')
#     parser.add_argument('--country_code_path',
#                         help='Path to the file containing the list of country codes to scrape, by default will use country_codes.txt in the same directory',
#                         default='country_codes.txt')
#     parser.add_argument('--data_dir', help='Path to save the outputted data in', default='data/')
#     parser.add_argument('--file_dir', help='Path to save the outputted files in', default='files/')
#     parser.add_argument('--model_dir', help='Path to file with model', default='model/')
#
#     args = parser.parse_args()
#
#     data_dir = args.data_dir
#     file_dir = args.file_dir
#     model_dir = args.model_dir
#
#     public_duration = 14
#     perform_predict()
#     perform_eda()
