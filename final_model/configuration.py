# configuration.py

import os

LANG_CODES = ["de", "en", "es", "fr", "it", "ko", "pt", "ta", "be", "ru"]
MAX_SENTENCES_PER_LANG = 50000
TEST_RATIO = 0.2
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLP_PATH = os.path.join(BASE_DIR, "data")

CONLLU_DATA_DIR = os.path.join(BASE_DIR, "preprocessing/data")
TWITTER_CSV_PATH = os.path.join(NLP_PATH, "tweets_dataset.csv")
STOPWORDS_PATH = os.path.join(NLP_PATH, "stopwords")