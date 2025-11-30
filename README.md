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

## Dataset Used
The **multilingual Wikipedia dataset** available on Hugging Face:  
https://huggingface.co/datasets/wikimedia/wikipedia

## Languages Selected
For this project, we selected **ten languages**, including both well-resourced and under-resourced languages. The chosen languages are:  

- Italian (it)  
- Tamil (ta)   
- Belarusian (be)   
- Russian (ru)  
- German (de)   
- Spanish (es)   
- English (en)   
- Portuguese (pt)   
- French (fr)   
- Korean (ko)   

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

### Potential issues

1. **Data Bias (Sample Size Differences)**  
   - The amount of available data varies significantly across languages due to differences in resource availability.  
   - As a result, some languages (e.g., Tamil, Belarusian) have fewer samples compared to high-resource languages like English or Spanish.  

2. **Tokenization Accuracy**  
   - The performance of Stanza’s tokenizer differs slightly across languages.  
   - For morphologically complex languages, tokenization errors may occur more frequently, which can affect downstream model performance.  

3. **Language Characters**  
   - Most of the selected languages use Latin-based scripts, while Russian and Belarusian share the Cyrillic script.  
   - This overlap may lead to misclassification between linguistically similar languages.


4. **Large-Scale Data Handling**  
   - Each preprocessed dataset per language can reach several gigabytes in size.  
   - Training and storing models on such large files may lead to memory limitations and long processing times, especially on standard hardware.  
   - Efficient data streaming, chunking, and storage optimizations (e.g., incremental loading, compressed formats, selective preprocessing) may be required to ensure model development.


## Milestone 1 – Mulitple Baseline Solutions

The mulitple baseline solutions include using both machine learning models and rule based models. It is then evaluated both quantitatively and qualitatively. 

### Machine learning model

1. **Data Ingestion** - arsing text from custom CoNLL-U files

2. **Feature Engineering:** - Converting raw text into numerical features using TF-IDF

3. **Model Training:** - Training a Multinomial Naive Bayes classifier

4. **Evaluation:** - Assessing performance using standard metrics



