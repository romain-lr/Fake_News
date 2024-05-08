from flask import Flask, render_template, request, redirect, url_for
from fake_news import detect_fake_news
from predirandomforest import predict_fake_news
import pandas as pd
from translation import translate_english_to_french, translate_french_to_english

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # User input
        user_input = request.form['user_input']
        
        # Import the training set        
        training_set = pd.read_csv('train_fake_news.csv')
        training_set.columns = ['A', 'B', 'C']
        del training_set['A']

        user_input_fr = translate_english_to_french(user_input) # equals user_input if the input is already in French
        user_input_en = translate_french_to_english(user_input) # equals user_input if the input is already in English
        result1 = detect_fake_news(user_input_en, training_set)
        result2 = predict_fake_news(user_input_en)
        return render_template('result.html', user_input=user_input_fr, result1=result1, result2=result2)

if __name__ == '__main__':
    app.run(debug=True)

