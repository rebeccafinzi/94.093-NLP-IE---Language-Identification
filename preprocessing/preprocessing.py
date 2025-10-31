import os
import re
from datasets import load_dataset
import stanza
from stanza.utils.conll import CoNLL
import argparse

PUNCT = r"\.,;:!\?\-\(\)\"'«»“”‘’…¿¡/%"

def clean_text_pt(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç0-9\s{PUNCT}]', '', text) # Keep only Portuguese characters
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_ko(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr"[^가-힣0-9\s{PUNCT}]", "", text) # Keep only Korean chars, digits, and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_fr(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-zÀÂÆÇÉÈÊËÎÏÔŒÙÛÜàâæçéèêëîïôœùûüÿ0-9\s{PUNCT}]', '', text) # Keep French characters and frequent used symbol
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_ru(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    # Keep only Russian characters (А-Я, а-я, Ё/ё), digits, spaces, frequent used symbol
    text = re.sub(fr"[^А-Яа-яЁё0-9\s{PUNCT}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_be(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    # Keep only Belarusian characters (А-Я, а-я, Ё/ё, І/і, Ў/ў), digits, spaces, frequent used symbol
    text = re.sub(fr"[^А-Яа-яЁёІіЎў0-9\s{PUNCT}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_it(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-zÀÈÉÌÒÙàèéìòù0-9\s{PUNCT}]', '', text)  # Keep Italian characters
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_es(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\s{PUNCT}]', '', text)  # Keep Spanish characters
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_en(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-z0-9\s{PUNCT}]', '', text)  # Keep English characters
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def process(lang, sample_size=0):
    try:
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", streaming=True)
        dataset_iter = dataset["train"]
    except Exception as e:
        print(f"[{lang}] Data Load Fail: {e}")
        return None

    clean_func = globals().get(f"clean_text_{lang}")

    stanza.download(lang, verbose=False)
    nlp = stanza.Pipeline(lang=lang, processors="tokenize")

    os.makedirs("data", exist_ok=True)
    output_path = f"data/output_{lang}.conllu"

    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, sample in enumerate(dataset_iter):
            if sample_size > 0 and i >= sample_size:
                break

            text = sample.get("text", "")
            if not text.strip():
                continue

            clean_text = clean_func(text)
            if not clean_text:
                continue

            doc = nlp(clean_text)
            CoNLL.write_doc2conll(doc, f_out)
            f_out.write("\n")

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="text preprocessing")
    parser.add_argument(
        "-lang",
        "--language",
        required=True,
        help="e.g. en, de, it, es, ko, "
    )
    parser.add_argument(
        "-n",
        "--sample_size",
        type=int,
        default=0,
        help="default: 0"
    )

    args = parser.parse_args()
    process(lang=args.language, sample_size=args.sample_size)