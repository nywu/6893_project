from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import numpy as np
import json
import pickle
import requests, sys, time, os
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


default_args = {
    'owner': 'Rita',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

api_key_path = '/home/nw2533/airflow/dags/api_key.txt'
country_code_path = '/home/nw2533/airflow/dags/country_codes.txt'
data_dir = '/home/nw2533/airflow/dags/data/'  # store extracted data
file_dir = '/home/nw2533/airflow/dags/file/'  # store output html and other files
model_dir = '/home/nw2533/airflow/dags/model/'  # store cluster & regression model and json

public_duration = 7

snippet_features = ["title", "publishedAt", "channelId", "channelTitle", "categoryId"]
unsafe_characters = ['\n', '"']
header = ["video_id"] + snippet_features + ["trending_date", "tags", "view_count", "likes", "dislikes",
                                           "comment_count", "thumbnail_link", "comments_disabled",
                                           "ratings_disabled", "description"]


def setup(api_path, code_path):
    with open(api_path, 'r') as file:
        api_key = file.readline()

    with open(code_path) as file:
        country_codes = [x.rstrip() for x in file]

    return api_key, country_codes


def prepare_feature(feature):
    for ch in unsafe_characters:
        feature = str(feature).replace(ch, "")
    return f'"{feature}"'


def api_request(page_token, country_code, api_key):
    request_url = f"https://www.googleapis.com/youtube/v3/videos?part=id,statistics,snippet{page_token}chart=mostPopular&regionCode={country_code}&maxResults=50&key={api_key}"
    request = requests.get(request_url)
    if request.status_code == 429:
        print("Temp-Banned due to excess requests, please wait and continue later")
        sys.exit()
    return request.json()


def get_tags(tags_list):
    return prepare_feature("|".join(tags_list))


def get_videos(items):
    lines = []
    for video in items:
        comments_disabled = False
        ratings_disabled = False

        if "statistics" not in video:
            continue

        video_id = prepare_feature(video['id'])
        snippet = video['snippet']
        statistics = video['statistics']
        features = [prepare_feature(snippet.get(feature, "")) for feature in snippet_features]

        description = snippet.get("description", "")
        thumbnail_link = snippet.get("thumbnails", dict()).get("default", dict()).get("url", "")
        trending_date = time.strftime("%y.%d.%m")
        tags = get_tags(snippet.get("tags", ["[none]"]))
        view_count = statistics.get("viewCount", 0)

        if 'likeCount' in statistics and 'favoriteCount' in statistics:
            likes = statistics['likeCount']
            dislikes = statistics['favoriteCount']
        else:
            ratings_disabled = True
            likes = 0
            dislikes = 0

        if 'commentCount' in statistics:
            comment_count = statistics['commentCount']
        else:
            comments_disabled = True
            comment_count = 0

        line = [video_id] + features + [prepare_feature(x) for x in [trending_date, tags, view_count, likes, dislikes,
                                                                     comment_count, thumbnail_link, comments_disabled,
                                                                     ratings_disabled, description]]
        lines.append(",".join(line))
    return lines


def get_pages(country_code, api_key, next_page_token="&"):
    country_data = []
    while next_page_token is not None:
        video_data_page = api_request(next_page_token, country_code, api_key)
        next_page_token = video_data_page.get("nextPageToken", None)
        next_page_token = f"&pageToken={next_page_token}&" if next_page_token is not None else next_page_token
        items = video_data_page.get('items', [])
        country_data += get_videos(items)
    return country_data


def write_to_file(country_code, country_data):
    print(f"Writing {country_code} data to file...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(f"{data_dir}/{country_code}_videos.csv", "w+", encoding='utf-8') as file:
        for row in country_data:
            file.write(f"{row}\n")


def get_data(api_key, country_codes):
    for country_code in country_codes:
        country_data = [",".join(header)] + get_pages(country_code, api_key)
        write_to_file(country_code, country_data)


def extract_data(**kwargs):
    api_key, country_codes = setup(api_key_path, country_code_path)
    get_data(api_key, country_codes)


def perform_predict(**kwargs):
    # List all CSV files in the specified directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith(('US_', 'CA_', 'FR_', 'IN_'))]
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

        for index, row in top_10_videos.iterrows():
            top_10_videos.loc[index, 'cluster'] = cluster_tags(selected_region, row)

        # prediction
        processed_data = process_data(top_10_videos)
        top_10_videos['Future_Likes'] = (make_predictions(processed_data, selected_region)).astype(int)
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


def perform_eda(**kwargs):
    # List all CSV files in the specified directory
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
        # videos = videos[videos['tags'] != '[none]']

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


def cluster_tags(selected_region, raw_data):
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


def process_data(raw_data):
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


def make_predictions(df, selected_region):
    # one-hot
    # category
    df['category_id'] = df['category_id'].astype(str)
    encoded_data = pd.get_dummies(df, columns=['category_id'], prefix='category')

    # load json
    file_name = None
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


with DAG(
    'youtube_data_extraction_and_prediction_new',
    default_args=default_args,
    description='YouTube Data Extraction DAG',
    schedule_interval=timedelta(minutes=1),
    catchup=False,
) as dag:
    extract_data_task = PythonOperator(
        task_id='extract_data_task',
        python_callable=extract_data,
        provide_context=True,
    )
    predict_task = PythonOperator(
        task_id='perform_predict',
        python_callable=perform_predict,
        provide_context=True,
        dag=dag,
    )
    eda_task = PythonOperator(
        task_id='perform_eda',
        python_callable=perform_eda,
        provide_context=True,
        dag=dag,
    )

# Set task dependencies
extract_data_task >> predict_task
extract_data_task >> eda_task