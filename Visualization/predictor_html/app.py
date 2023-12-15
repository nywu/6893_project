from flask import Flask, request, render_template
import model_utils
import langid
from googletrans import Translator, LANGUAGES

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    selected_region = request.form.get('selected_region')
    time_selection = request.form.get('time_selection')
    title = request.form.get('title')
    description = request.form.get('description')
    tag = request.form.get('tag')

    
    target_lang='en'
    detected_lang1 = langid.classify(title)[0]    
    if detected_lang1 != target_lang:
        title= model_utils.translate_text(title)
       
    detected_lang2 = langid.classify(description)[0]    
    if detected_lang2 != target_lang:
        description= model_utils.translate_text(description)
        
    detected_lang3 = langid.classify(tag)[0]    
    if detected_lang3 != target_lang:
        tag= model_utils.translate_text(tag)


    category = request.form.get('category')
    view_count = int(request.form.get('view_count', 0)) if request.form.get('view_count') != '' else 0
    comment_count = int(request.form.get('comment_count', 0)) if request.form.get('comment_count') != '' else 0

    if title == '' or description == '' or tag == '':
        error_message = 'Title, description and tags are required.'
        return render_template('index.html', error_message=error_message)

    processed_data = model_utils.process_data(selected_region, view_count, time_selection, comment_count, title, description, tag, category)
    prediction = model_utils.make_predictions(processed_data, selected_region)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
