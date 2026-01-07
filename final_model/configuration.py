# configuration.py

LANG_CODES = ["de", "en", "es", "fr", "it", "ko", "pt", "ta", "be", "ru"]
MAX_SENTENCES_PER_LANG = 50000
TEST_RATIO = 0.2
RANDOM_SEED = 42

NLP_PATH= "/content/drive/MyDrive/NLP"

CONLLU_DATA_DIR = f"{NLP_PATH}/preprocessing/data"
TWITTER_CSV_PATH = f"{NLP_PATH}/final_solution/data/tweets_dataset.csv"
STOPWORDS_PATH = f"{NLP_PATH}/final_solution/data/stopwords"