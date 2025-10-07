# translate.py
from googletrans import Translator
from langdetect import detect

translator = Translator()

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "en"

def to_english(text):
    try:
        # if already english, return
        if detect_lang(text).startswith("en"):
            return text
        return translator.translate(text, src='auto', dest='en').text
    except Exception as e:
        # Fallback if translation fails
        return f"[Translation Error] {text}"

def to_urdu(text):
    try:
        return translator.translate(text, src='en', dest='ur').text
    except Exception as e:
        # Fallback if translation fails
        return f"[Translation Error] {text}"