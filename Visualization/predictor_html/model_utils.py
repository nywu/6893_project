import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator

translator = Translator()
def translate_text(text):
    try:
        return translator.translate(text, dest='en').text
    except:
        return text  # Return original text if translation fails


def process_data(selected_region, view_count, time, comment_count, title, description, tags, categoryID):

    # log
    views = np.log1p(view_count)

    comment_count = np.log1p(comment_count)

    # get tag cluster
    tags_list = tags.split(',')
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
    cwd = os.getcwd()
    model_path = os.path.join(cwd, 'models', f'{selected_region}_kmeans_model.pkl')
    kmeans = pd.read_pickle(model_path)



    # filename = f'/models/{selected_region}_kmeans_model.pkl'
    # with open(filename, 'rb') as file:
    #     kmeans = pickle.load(file)
    # get cluster
    features_array = features_array.reshape(1, -1)
    cluster = kmeans.predict(features_array)[0]

    csv_path = os.path.join(cwd, 'models', f'{selected_region}_means.csv')
    df = pd.read_csv(csv_path)

    # get necessary value from kmeans
    df_new = df[df['cluster'] == cluster][['dislikes']]
    dislikes=df_new['dislikes'].values[0]
    dislikes = np.log1p(dislikes)

    # sentiment
    def analyze_sentiment(text):
        blob = TextBlob(str(text))
        sentiment = blob.sentiment.polarity
        return sentiment

    title_sentiment = analyze_sentiment(title)
    description_sentiment = analyze_sentiment(description)

    scaler = StandardScaler()
    normalized_total_published_time = scaler.fit_transform([[time]]).item()

    # return a preprocessed dataframe with tag cluster
    data = [categoryID, views, normalized_total_published_time, title_sentiment,
            description_sentiment, comment_count, dislikes, cluster]
    columns = ['category_id', 'views', 'normalized_total_published_time',
               'title_sentiment', 'description_sentiment', 'comment_count',
               'dislikes', 'cluster']
    datadf = pd.DataFrame([data], columns=columns)

    return datadf



def make_predictions(df, selected_region):
    # one-hot
    # category
    encoded_data = pd.get_dummies(df, columns=['category_id'], prefix='category')

    # fill category id
    import json
    region = selected_region
    file_name = region + '_category_id.json'
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'models', file_name)
    if file_name is not None:
        with open(file_path, 'r') as f:
            data = json.load(f)
            max_id = max(int(category['id']) for category in data['items'])
    else:
        print(f"No JSON file found for region: {region}")

    all_categories = range(1, max_id + 1)
    for category in all_categories:
        column_name = f'category_{category}'
        if column_name not in encoded_data.columns:
            encoded_data[column_name] = 0

    # tag
    data_to_predict = pd.get_dummies(encoded_data, columns=['cluster'], prefix='cluster')

    for cluster in range(0, 20):
        column_name = f'cluster_{cluster}'
        if column_name not in data_to_predict.columns:
            data_to_predict[column_name] = 0

    print(data_to_predict)


    model_path = os.path.join(cwd, 'models', f'{selected_region}model.pkl')
    model = pd.read_pickle(model_path)

    # model_path = f'/models/{selected_region}model.pkl'
    # with open(model_path, 'rb') as model_file:
    #     model = pickle.load(model_file)

    feature_order = model.feature_names_in_
    data_to_predict = data_to_predict[feature_order]
    predictions = model.predict(data_to_predict)
    likes = np.expm1(predictions)

    return np.round(likes)


