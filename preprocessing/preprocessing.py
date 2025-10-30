import os
import re
import pandas as pd
from datasets import load_dataset
import stanza
from stanza.utils.conll import CoNLL
import argparse

def clean_text_pt(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'[^A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç0-9\s.]', '', text) # Keep only Portuguese characters
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_ko(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r"[^가-힣0-9\s.]", "", text) # Keep only Korean chars, digits, and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_fr(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'[^A-Za-zÀÂÆÇÉÈÊËÎÏÔŒÙÛÜàâæçéèêëîïôœùûüÿ0-9\s\'\-.]', '', text) # Keep French characters and frequent used symbol
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_ru(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    # Keep only Russian characters (А-Я, а-я, Ё/ё), digits, spaces, frequent used symbol
    text = re.sub(r"[^А-Яа-яЁё0-9\s\.\-’ʼ']", "", text)
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_be(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    # Keep only Belarusian characters (А-Я, а-я, Ё/ё, І/і, Ў/ў), digits, spaces, frequent used symbol
    text = re.sub(r"[^А-Яа-яЁёІіЎў0-9\s\.\-’ʼ']", "", text)
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def process_and_save_conllu(df, lang, batch_size=50, processors="tokenize,pos,lemma", output_path=None):
    texts = [t for t in df["clean_text"].astype(str).tolist() if t.strip()]
    
    stanza.download(lang, verbose=False)
    nlp = stanza.Pipeline(lang=lang, processors=processors)

    os.makedirs("data", exist_ok=True)
    if output_path is None:
        output_path = f"data/output_{lang}.conllu"

    with open(output_path, "w", encoding="utf-8") as f:
        pass

    for i in range(0, len(texts), batch_size):
        for t in texts[i:i+batch_size]:
            doc = nlp(t)
            with open(output_path, "a", encoding="utf-8") as f:
                CoNLL.write_doc2conll(doc, f)
                f.write("\n")
                
    return output_path

def process(lang, sample_size=1000):
    try:
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}")
    except Exception as e:
        print(f"[{lang}] Data Load Fail: {e}")
        return None

    df = pd.DataFrame(dataset["train"])
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    clean_func = globals().get(f"clean_text_{lang}")
    df["clean_text"] = df["text"].apply(clean_func)
    
    output_path = process_and_save_conllu(df, lang=lang)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="text preprocessing")
    parser.add_argument(
        "-lang",
        "--language",
        required=True,
        help="e.g. en, de, it, es, ko, ru, be"
    )
    parser.add_argument(
        "-n",
        "--sample_size",
        type=int,
        default=1000,
        help="default: 1000"
    )

    args = parser.parse_args()
    process(lang=args.language, sample_size=args.sample_size)