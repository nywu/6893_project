import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Load the dataset
# file = 'gs://6893_w/CAvideos.csv'
# load tag modified csv
file='USA_tags_modified.csv'

def split_data(file_name):
    # load data
    df = pd.read_csv(file_name)

    # Feature selection
    features = ['category_id','normalized_total_published_time', 'title_sentiment', 'description_sentiment', 'views',
                'comment_count', 'dislikes','cluster']

    # Extracting features
    X = df[features]

    # Target variable
    y = df['likes']
    y_log = np.log1p(y)

    # one-hot
    # category
    encoded_data = pd.get_dummies(X, columns=['category_id'], prefix='category')

    # fill category id
    import json

    replacement_dict = {
        'Canada': 'CA',
        'Germany': 'DE',
        'France': 'FR',
        'Great Britain': 'GB',
        'India': 'IN',
        'Mexico': 'MX',
        'South Korea': 'KR',
        'Japan': 'JP',
        'Russia': 'RU',
        'USA': 'US'
    }

    region = df['region'][0]

    # load json
    file_name = None
    for key, value in replacement_dict.items():
        if region == key:
            file_name = value + '_category_id.json'
            break

    if file_name is not None:
        with open(file_name, 'r') as f:
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
    X_encoded = pd.get_dummies(encoded_data, columns=['cluster'], prefix='cluster')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_log, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, region
def select_best_model(X_train, y_train, X_test, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        # "Ridge": Ridge(),
        # "Lasso": Lasso()
    }

    results = {}

    # train models
    for name, model in models.items():
        # train
        model.fit(X_train, y_train)
        # prediction
        y_pred = model.predict(X_test)
        # evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = (mse, r2)

    # select the best one
    best_model_name = min(results, key=lambda x: results[x][0])  # based on mse
    best_model_instance = models[best_model_name]

    return best_model_instance, results[best_model_name]

X_train, X_test, y_train, y_test, region = split_data(file)
print(region)

model, best_model_performance = select_best_model(X_train, y_train, X_test, y_test)
print(f'Best Model Performance: MSE: {best_model_performance[0]}, R2: {best_model_performance[1]}')

# print feature importance
column_names = X_train.columns
feature_names = column_names
feature_importances = model.feature_importances_

# create importance DataFrame
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
# sort
importance_df = importance_df.sort_values(by='Importance', ascending=False)

for index, row in importance_df.iterrows():
    feature_name = row['Feature']
    importance = row['Importance']
    print(f"Feature: {feature_name}, Importance: {importance}")


import pickle
model_path = f'{region}model.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

# save model to google cloud
# import pickle
#
# model_filename = f'{region}model.pkl'
# with open(model_filename, 'wb') as file:
#     pickle.dump(model, file)
#
# from google.cloud import storage
# import os
#
# client = storage.Client()
# bucket_name = '6893_w'
#
# bucket = client.bucket(bucket_name)
#
# filename = f'{region}model.pkl'
# blob = bucket.blob(filename)
# blob.upload_from_filename(filename)


