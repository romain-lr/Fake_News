from googletrans import Translator, LANGUAGES

def translate_english_to_french(text):
    # Create an instance of the translator
    translator = Translator()

    # Detect the language of the original text (English in this case)
    detected_lang = translator.detect(text).lang

    # If the detected language is English, translate to French
    if detected_lang == 'en':
        translation = translator.translate(text, src='en', dest='fr')
        return translation.text
    else:
        # If the detected language is not English, return the original text
        return text

def translate_french_to_english(text):
    # Create an instance of the translator
    translator = Translator()

    # Detect the language of the original text (french in this case)
    detected_lang = translator.detect(text).lang

    # If the detected language is French, translate to English
    if detected_lang == 'fr':
        translation = translator.translate(text, src='fr', dest='en')
        return translation.text
    else:
        # If the detected language is not French, return the original text
        return text
