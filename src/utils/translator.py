from transformers import pipeline

_translate_pipeline = None


def _get_translator():
    global _translate_pipeline
    if _translate_pipeline is None:
        # Helsinki-NLP model for English->Urdu
        _translate_pipeline = pipeline('translation', model='Helsinki-NLP/opus-mt-en-ur')
    return _translate_pipeline


def translate_to_urdu(text: str) -> str:
    translator = _get_translator()
    out = translator(text, max_length=400)
    if isinstance(out, list):
        return out[0]['translation_text']
    return str(out)
