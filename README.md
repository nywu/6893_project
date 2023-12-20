Trending YouTube - Analysis and Prediction

Help video creators better evaluate video descriptions when posting videos.
Assist platforms in evaluating and adjusting video exposure.


Steps for visualization:
1. Open a terminal or command prompt in project directory.
2. Install the required server tools, such as npm install -g http-server to install http-server. In a terminal or command prompt, navigate to project directory.
3. Run the command http-server -c-1, and it will start a local server, run python3 app.py for predictor.
4. Visit http://localhost:8080 or the corresponding server address in the browser.
5. Execute python app.py in the path .\Visualization\predictor_html\predicthtml

Steps for Airflow:
1. Set up Airflow environment and initialize the Airflow database
2. Start the Airflow scheduler
3. Upload files in the specified directory structure
```markdown
├── dags/
│   ├── scrap_predict_new.py
│   ├── country_code.txt
│   ├── api_key.txt
│   ├── model/
│   │   ├── category_id.json
│   │   ├── predict_model.pkl
│   │   └── kmeans_model.pkl
```
5. Execute the Python script
```bash
python ~/airflow/dags/scrap_predict_new.py
```
7. Trigger the DAG in Airflow
