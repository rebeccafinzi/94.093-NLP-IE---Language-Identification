# 94.093-NLP&IE-Language Identification

**Group:** Linguisticglitch  
**Members:**  
- Olesia Galynskaia (12321492)  
- Julia Chalissery (52007582)  
- Yeongshin Park (12432938)  
- Rebecca Micol Finzi (12505266)
  
---

## Project Overview
The objective of this project is to build a **multilingual dataset for automatic language identification**. Language identification is a fundamental NLP task that involves automatically determining the language of a given text segment.

## Languages Selected
For this project, we selected **ten languages**, including both well-resourced and under-resourced languages. The chosen languages are:  

- Italian  
- Tamil  
- Belarusian  
- Russian  
- German  
- Spanish  
- English  
- Portuguese  
- French  
- Korean  

---

## Milestone 1 – Preprocessing

The preprocessing pipeline was designed to produce a **clean, consistent, and standardized multilingual dataset**. The following steps were applied:

1. **Removal of unwanted characters** – Non-language-specific characters, extra symbols, and unnecessary whitespace were removed. Each language has a tailored cleaning function to retain only valid characters for that language.
   
2. **Whitespace normalization** – Multiple consecutive spaces and newlines were replaced with a single space.

3. **Tokenization** – Text was tokenized using **Stanza**, a state-of-the-art NLP library that respects language-specific rules.

4. **Saving in CoNLL format** – Finally, the cleaned and tokenized text was saved in CoNLL-U format, suitable for NLP model training and evaluation.

Stopword removal, stemming, and lemmatization are intentionally excluded.  
These operations can remove morphological or orthographic cues that are crucial for distinguishing languages in language identification tasks.

### Data and Configuration

- Each language is processed using 10,000 Wikipedia pages, chosen as a practical balance between computational cost (RAM, runtime, storage) and representativeness.

- Cleaned text retains only language-appropriate alphabets, digits, and the following punctuation marks:
```
PUNCT = r"\.,;:!\?\-\(\)\"'«»“”‘’…¿¡/%"
```

These punctuation symbols are preserved because their usage frequency and positioning differ across languages, making them valuable discriminative features.

- After processing, all files are saved in:
```
preprocessing/data/output_{lang}.conllu
```
However, since GitHub has file size limitations, the resulting .conllu files have been uploaded to Google Drive instead:
https://drive.google.com/drive/folders/1osun1pC_xVrVZmb-LnneVxrUtf-o_GvR

### Structure
```
94.093-NLP-IE---Language-Identification/
└── preprocessing/
    ├── data/
    │   ├── output_fr.conllu
    │   ├── output_ko.conllu
    │   ├── output_pt.conllu
    │   └── ...
    │
    ├── preprocessing.py
    ├── requirements.txt
    └── run.sh
```

### How to run
#### Option 1: Run All Languages Sequentially
```
chmod +x run.sh
./run.sh
```
#### Option 2: Run a Single Language Manually
```
python preprocessing.py -lang en -n 10000
```

   


