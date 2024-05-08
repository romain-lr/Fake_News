#MesTests

from fake_news import detect_fake_news  
from translation import translate_english_to_french, translate_french_to_english
import pandas as pd
from predirandomforest import predict_fake_news

filepath1 = "./train_fake_news.csv"

training_set = pd.read_csv(filepath1)
training_set.columns = ['A', 'B', 'C']
del training_set['A']

#Test fake news detection with a real news example.
def test_FN_true(): 
    text = "Ceci est une fake news."
    result = detect_fake_news(text,training_set)
    assert result == "Real News", f"Expected 'Real News' but got '{result}'."

#Test fake news detection with a fake news example.
def test_FN_false():
    text = "Hillary Clinton agrees with John McCain :by voting to give George Bush the benefit of the doubt on Iran."
    result = detect_fake_news(text,training_set)
    assert result == "Fake News", f"Expected 'Fake News' but got '{result}'."

#Test translation with a non-English text.
#def test_translation_en():
    #text = "Bonjour"
    #translated_text = translate_french_to_english(text)
    #assert translated_text == "Hello", f"Expected 'Hello' but got '{translated_text}'."

#Test translation with a non-French text.
#def test_translation_fr():
    #text = "Hello"
    #translated_text = translate_english_to_french(text)
    #assert translated_text == "Bonjour", f"Expected 'Bonjour' but got '{translated_text}'."

#Test predict fake news with random forest
#Test for a fake news
def test_FAKE():
    result = predict_fake_news("Donald Trump is against marriage equality. He wants to go back.")
    assert result == "Fake News", f"Expected 'Fake News' but got '{result}'."

 #Test for a real news   
def test_REAL():
    result = predict_fake_news("Denali is the Kenyan word for black power.")
    assert result == "Real News", f"Expected 'Real News' but got '{result}'."