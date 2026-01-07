# load_data.py

from .configuration import LANG_CODES, MAX_SENTENCES_PER_LANG, CONLLU_DATA_DIR, TWITTER_CSV_PATH
import os
import re
from tqdm import tqdm
import pandas as pd
from collections import Counter

def clean_text_basic(text):
    text = re.sub(r"https?://\S+", " ", text)

    cleaned_chars = []
    for ch in text:
        if ch.isalpha() or ch.isspace():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")

    text = "".join(cleaned_chars)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_conllu_sentences(path):
    sentences = []
    current_tokens = []
    current_text = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")

                if line.startswith("# text = "):
                    current_text = line.split("=", 1)[1].strip()
                elif not line:
                    if current_text:
                        cleaned = clean_text_basic(current_text)
                        if cleaned and len(cleaned) > 5:
                            sentences.append(cleaned)
                    elif current_tokens:
                        sent = " ".join(current_tokens)
                        cleaned = clean_text_basic(sent)
                        if cleaned and len(cleaned) > 5:
                            sentences.append(cleaned)
                    current_tokens = []
                    current_text = None
                elif line.startswith("#"):
                    continue
                else:
                    cols = line.split("\t")
                    if len(cols) >= 2:
                        current_tokens.append(cols[1])
    except FileNotFoundError:
        print(f"File not found: {path}")
        return []

    return sentences

def load_wikipedia_data(max_per_lang=MAX_SENTENCES_PER_LANG):
    texts = []
    labels = []

    for lang in tqdm(LANG_CODES):
        path = os.path.join(CONLLU_DATA_DIR, f"output_{lang}.conllu")
        sents = load_conllu_sentences(path)

        if max_per_lang and len(sents) > max_per_lang:
            sents = sents[:max_per_lang]

        texts.extend(sents)
        labels.extend([lang] * len(sents))
        print(f"  {lang}: {len(sents)} sentences")

    print(f"Total: {len(texts)} sentences")
    return texts, labels

def load_twitter_data():

    if not os.path.exists(TWITTER_CSV_PATH):
        print(f"File not found: {TWITTER_CSV_PATH}")
        return [], []

    df = pd.read_csv(TWITTER_CSV_PATH)

    TWITTER_LANG_MAPPING = {
        "en": "en", "de": "de", "es": "es", "fr": "fr",
        "it": "it", "ko": "ko", "pt": "pt", "ru": "ru", 
        "be": "be", "ta": "ta"
    }

    texts = []
    labels = []

    for _, row in df.iterrows():
        text = str(row.get("content", ""))
        lang = str(row.get("language", ""))

        if lang in TWITTER_LANG_MAPPING:
            cleaned = clean_text_basic(text)
            if cleaned and len(cleaned) > 5:
                texts.append(cleaned)
                labels.append(TWITTER_LANG_MAPPING[lang])

    print(f"Total: {len(texts)} tweets")
    print(f"Distribution: {Counter(labels)}")
    return texts, labels
