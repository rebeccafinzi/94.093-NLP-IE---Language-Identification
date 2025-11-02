import os
import re
from datasets import load_dataset
import stanza
from stanza.utils.conll import CoNLL
import argparse
from tqdm import tqdm 

'''
Define cleaning rules:
- Include punctuation reflecting language-specific usage
- Keep only language-native alphabets and commonly used symbols
- Retain digits
- Drop other scripts
'''

PUNCT = r"\.,;:!\?\-\(\)\"'«»“”‘’…¿¡/%"

def clean_text_pt(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇáàâãéêíóôõúç0-9\s{PUNCT}]', '', text) # Keep Portuguese characters and frequently used symbols
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_ko(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr"[^가-힣0-9\s{PUNCT}]", "", text) # Keep Korean characters and frequently used symbols
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_fr(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-zÀÂÆÇÉÈÊËÎÏÔŒÙÛÜàâæçéèêëîïôœùûüÿ0-9\s{PUNCT}]', '', text) # Keep French characters and frequently used symbols
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_ru(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    # Keep only Russian characters (А-Я, а-я, Ё/ё), digits, spaces, frequently used symbols
    text = re.sub(fr"[^А-Яа-яЁё0-9\s{PUNCT}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_be(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    # Keep only Belarusian characters (А-Я, а-я, Ё/ё, І/і, Ў/ў), digits, spaces, frequently used symbols
    text = re.sub(fr"[^А-Яа-яЁёІіЎў0-9\s{PUNCT}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_it(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-zÀÈÉÌÒÙàèéìòù0-9\s{PUNCT}]', '', text)  # Keep Italian characters and frequently used symbols
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_es(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\s{PUNCT}]', '', text)  # Keep Spanish characters and frequently used symbols
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_en(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(fr'[^A-Za-z0-9\s{PUNCT}]', '', text)  # Keep English characters and frequently used symbols
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def clean_text_de(text):
    text = re.sub(r'\n', ' ', text) # Remove newline characters
    text = re.sub(fr'[^A-Za-zÄÖÜäöüß0-9\s{PUNCT}]', '', text)  # Keep German characters and frequently used symbols
    text = re.sub(r'\s+', ' ', text).strip() # multiple spaces
    return text

def clean_text_ta(text):
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    # Keep Tamil characters (Unicode range \u0B80-\u0BFF) and frequently used symbols
    text = re.sub(fr"[^\u0B80-\u0BFF0-9\s{PUNCT}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()  # multiple spaces
    return text

def process(lang, sample_size=0):
    '''
    - Loads, cleans, and tokenizes Wikipedia text
    - Saves it in CoNLL-U format 
    '''
    
    # Load wikipedia dataset
    # Set streaming=True to avoid loading the entire dataset
    try:
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", streaming=True)
        dataset_iter = dataset["train"]
    except Exception as e:
        print(f"[{lang}] Data Load Fail: {e}")
        return None

    # Select clean_text_{lang} function
    clean_func = globals().get(f"clean_text_{lang}")
    if clean_func is None:
        print(f"[{lang}] No cleaning function found. Skipping language.")
        return None

    # Download and initialize Stanza tokenizer
    # Apply tokenization only
    stanza.download(lang, verbose=False)
    nlp = stanza.Pipeline(lang=lang, processors="tokenize")

    os.makedirs("data", exist_ok=True)
    output_path = f"data/output_{lang}.conllu"

    total = sample_size if sample_size > 0 else None
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, sample in enumerate(
            tqdm(dataset_iter, total=total, desc=f"[{lang}] Progress", unit="doc", mininterval=1.0)
        ):
            if sample_size > 0 and i >= sample_size:
                break
            
            # Choose text field from wikipedia dataset
            text = sample.get("text", "")
            if not text.strip():
                continue
            
            # Apply the clean_text function for the selected language
            clean_text = clean_func(text)
            if not clean_text:
                continue
            
            # Apply tokenization and save in CoNLL-U format
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
        help="e.g. en, de, it, es, ko, pt, fr, be, ru, ta "
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