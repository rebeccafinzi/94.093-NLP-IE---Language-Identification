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

## Structure
```
94.093-NLP-IE---Language-Identification/
├── data/
│   ├── tweets_dataset.csv
│   └── stopwords/
│       ├── be.txt
│       ├── de.txt
│       ├── en.txt
│       └── ...
│
├── preprocessing/
│   ├── data/
│   │   ├── output_fr.conllu
│   │   ├── output_ko.conllu
│   │   ├── output_pt.conllu
│   │   └── ...
│   │
│   ├── preprocessing.py
│   ├── requirements.txt
│   ├── run.sh
│   └── README.md
│
├── baseline_submission_ml.py
├── baseline_submission_ml.ipynb
├── rule_based_langid.py
├── rule_based_langid_param.ipynb
│
├── final_model/
│   ├── __init__.py
│   ├── configuration.py
│   ├── load_data.py
│   ├── rulebased.py
│   ├── finetuned_xlmroberta.py
│   └── main.py
│ 
├── model_evaluation.ipynb
└── README.md

```

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

---

## Milestone 2 – Mulitple Baseline Solutions

The mulitple baseline solutions include using both machine learning models and rule based models. It is then evaluated both quantitatively and qualitatively. 

### Structure
```
94.093-NLP-IE---Language-Identification/
└── preprocessing/
│   ├── data/
│   │   ├── output_fr.conllu
│   │   ├── output_ko.conllu
│   │   ├── output_pt.conllu
│   │   └── ...
│   │
│   ├── preprocessing.py
│   ├── requirements.txt
│   └── run.sh
│
├── rule_based_langid.py - final rule-based model
├── rule_based_langid_param.ipynb - rule-based model process of parameters choosing with all the results printed
├── baseline_submission_ml.py - final ML-based model
├── baseline_submission_ml.ipynb - final ML-based model with all the results printed
└── README.md
```

---

### Rule-based model

The rule-based system for language identification was built on script detection, stopwords, character-level n-grams (bi-, tri-, four-grams), and language-specific special characters.  
The method does not require ML model training. Instead, it extracts statistical frequency patterns from the multilingual corpus and stores them inside rule sets for each language.  
The language is chosen either by alphabet (if it's unique in the language group) or by scoring.  

#### Methodology

**1. Script Detection**  
Each input text is inspected for alphabet family:  
- Latin (en, de, fr, es, it, pt)  
- Cyrillic (ru, be)  
- Hangul (ko)  
- Tamil (ta)  

If only one script is present, classification is restricted to languages using that script.  
If multiple scripts are mixed, all languages are considered.

This step significantly reduces false positives.

**2. Character-level N-grams**  
For every language, the system extracts the top *K* most frequent:
- bigrams
- trigrams
- four-grams

These n-grams capture stable orthographic patterns unique to each language.

Weights applied during scoring:  

| Feature | Weight |
|--------|--------|
| bigram | +1 |
| trigram | +2 |
| four-gram | +3 |

**3. Stopwords & Special Characters**

The list of stopwords (unique but quite freaquent words for the language) and special characters was made manually based on the information on the internet.

| Feature | Weight |
|---------|--------|
| stopword | +3 |
| language-specific characters | +4 |

These features help disambiguate languages that share scripts or similar vocabulary.

**4. Language Prediction**  
- If script filtering results in exactly one candidate language - it is returned immediately.  
- Otherwise all candidates receive a rule-based score.  
- The language with the highest score is predicted.  
- If all scores equal zero the result is `unknown`.

#### Hyperparameter Search

Due to data volume restrictions and noticing, that more data doesn't mean better results, samples of different sizes for each language were tried.  
Other parameters were the number of top n-grams and scoring schemes.

The following weight configurations were used during the experiments:

| Feature                 | `base` | `strong_ngrams` | `strong_special` |
|------------------------|:------:|:---------------:|:----------------:|
| stopwords              |   3.0  |       2.0       |       2.0        |
| language-specific char |   4.0  |       3.0       |       6.0        |
| bigram match           |   1.0  |       1.5       |       1.0        |
| trigram match          |   2.0  |       3.0       |       2.0        |
| four-gram match        |   3.0  |       4.0       |       3.0        |

`base` is the balanced configuration used in the final rule-based model.  
`strong_ngrams` increases the contribution of all n-grams.  
`strong_special` heavily increases the influence of language-specific characters.

A large-scale evaluation tested combinations of:

| Parameter | Tested Values |
|----------|----------------|
| max_sent_per_lang | 50k, 100k, 150k, 200k |
| n-gram top_k | 20, 30, 40 |
| scoring schemes | base, strong_ngrams, strong_special |

Total: 36 configurations.

#### Results

The rule-based model was evaluated using **overall accuracy**, computed as the ratio of correctly predicted sentences to the total number of sentences in the test set.  
Accuracy was chosen as the final evaluation metric because the task is a single-label, closed-set classification problem with balanced class distribution after sampling.  
Other metrics (precision, recall, F1) are less informative here, since the rule-based system does not output probabilistic predictions and cannot be tuned per class.

**Accuracy across configurations:**

| max_sent_per_lang | top_k | weights         | accuracy |
|-------------------:|------:|-----------------|----------:|
| 50000              | 40    | base            | 0.939160 |
| 150000             | 40    | base            | 0.937730 |
| 100000             | 40    | base            | 0.937525 |
| 50000              | 40    | strong_ngrams   | 0.935700 |
| 50000              | 40    | strong_special  | 0.934950 |
| 100000             | 40    | strong_ngrams   | 0.934245 |
| 50000              | 30    | base            | 0.934190 |
| 150000             | 40    | strong_ngrams   | 0.934173 |
| 150000             | 40    | strong_special  | 0.933093 |
| 100000             | 40    | strong_special  | 0.932765 |
| 200000             | 40    | base            | 0.932250 |
| 150000             | 30    | base            | 0.931037 |
| 50000              | 30    | strong_ngrams   | 0.930690 |
| 100000             | 30    | base            | 0.930215 |
| 50000              | 30    | strong_special  | 0.929510 |
| 200000             | 40    | strong_ngrams   | 0.928747 |
| 200000             | 40    | strong_special  | 0.927662 |
| 200000             | 30    | base            | 0.927325 |
| 150000             | 30    | strong_ngrams   | 0.926940 |
| 50000              | 20    | base            | 0.926760 |
| 100000             | 30    | strong_ngrams   | 0.926665 |
| 150000             | 30    | strong_special  | 0.926077 |
| 100000             | 30    | strong_special  | 0.924830 |
| 200000             | 30    | strong_ngrams   | 0.923273 |
| 200000             | 30    | strong_special  | 0.922095 |
| 150000             | 20    | base            | 0.921300 |
| 100000             | 20    | base            | 0.921245 |
| 50000              | 20    | strong_ngrams   | 0.921190 |
| 50000              | 20    | strong_special  | 0.921070 |
| 100000             | 20    | strong_ngrams   | 0.916715 |
| 200000             | 20    | base            | 0.916510 |
| 150000             | 20    | strong_ngrams   | 0.916057 |
| 100000             | 20    | strong_special  | 0.914860 |
| 150000             | 20    | strong_special  | 0.914443 |
| 200000             | 20    | strong_ngrams   | 0.911680 |
| 200000             | 20    | strong_special  | 0.910743 |

Accuracy converges near 91–94%.  
Increasing max_sent beyond 50k or top_k beyond 40 does not improve performance.

#### Final Chosen Configuration

- **max_sent_per_lang = 50,000**  
- **top_k_n_grams = 40**  
- **weights = base**  
- **accuracy = 0.9392**

This is the most stable configuration with the best balance of accuracy and runtime ratio.

#### Confusions

| gold → predicted | count |
|------------------|-------|
| be → ru | 809 |
| es → pt | 725 |
| es → en | 339 |
| de → en | 337 |
| pt → fr | 297 |
| pt → es | 276 |
| es → it | 249 |
| pt → en | 214 |
| es → fr | 200 |
| ru → be | 162 |
| fr → en | 155 |
| it → es | 154 |
| it → en | 152 |
| es → de | 151 |

Most errors reflect real linguistic similarity:
- Belarusian and Russian (shared Cyrillic script and same words)
- Romance languages (es, pt, it, fr)
- Occasional errors from lexical borrowing

---

### Machine learning model

1. **Data Ingestion** - parsing text from custom CoNLL-U files

2. **Feature Engineering:** - Converting raw text into numerical features using TF-IDF

3. **Model Training:** - Training a Multinomial Naive Bayes classifier

4. **Evaluation:** - Assessing performance using standard metrics

#### Methodology

1. **Data Processing**
   - Text data is extracted from CoNLL-U formatted files (specifically targeting the `# text = ` field) located in the `data/` directory. A total of **6,669,377 sentences**      were loaded and used for training and testing. The dataset was split into training and testing sets with an **80/20 ratio**, using **stratified sampling** to ensure         the proportion of each language is maintained in both sets.

2. **Feature Engineering**
   - **TF-IDF (Term Frequency-Inverse Document Frequency)** was used to weigh the importance of words (unigrams) in the corpus. The vectorizer was limited to the **top           1,000 most frequent features** to maintain efficiency and generalization across a large, multilingual vocabulary.

3. **Model Selection**
   - **Naive Bayes** was chosen. MNB is a well-suited baseline for text classification tasks, particularly when using count or TF-IDF features, due to its simplicity,            speed, and strong performance.

#### Performance
| Metric | Value |
| :--- | :--- |
| **Test Set Size** | 1,333,876 |
| **Model Accuracy** | **0.8727 (87.27%)** |

The model was evaluated on the 20% test set, comprising **1,333,876 sentences**. The model shows excellent performance across most Latin-script languages (e.g., German, English, Spanish) but reveals specific challenges with languages that have lower training sample counts or share significant vocabulary with other languages.

| Language | Precision | Recall | F1-Score | Support | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ta** (Tamil) | 1.00 | 0.99 | 1.00 | 44,321 | Near-perfect performance. |
| **de** (German) | 0.98 | 0.95 | 0.96 | 133,047 | Very strong performance. |
| **it** (Italian) | 0.99 | 0.94 | 0.96 | 67,156 | Excellent discrimination. |
| **en** (English) | 0.95 | 0.97 | 0.96 | 123,567 | High recall and precision. |
| **es** (Spanish) | 0.92 | 0.94 | 0.93 | 202,086 | Strong, reliable performance. |
| **fr** (French) | 0.98 | 0.88 | 0.93 | 133,289 | High precision, but some false negatives (lower recall). |
| **pt** (Portuguese) | 0.99 | 0.83 | 0.90 | 145,715 | High precision, but lower recall suggests confusion with related languages (e.g., Spanish, French). |
| **ru** (Russian) | 0.72 | 1.00 | 0.83 | 351,998 | **High Recall (1.00):** Captures almost all Russian sentences, but **low Precision (0.72)** indicates it frequently misclassifies other languages as Russian (false positives). |
| **be** (Belarusian) | 0.92 | 0.25 | 0.39 | 51,368 | **Low Recall:** Suggests high confusion with Russian due to shared Cyrillic script and potentially low sample diversity. |
| **ko** (Korean) | 0.99 | 0.24 | 0.39 | 81,329 | **Low Recall:** Indicates many Korean sentences are misclassified, despite a unique script. Potential issues with feature representation or sparsity. |

#### Training with SVC model 
SVC was attempted for comparison, but encountered severe scalability limitations. Training the SVC model on the full 5.3 million sample set proved infeasible. The process was terminated after running for over 48 hours without completion, confirming that SVC is not a practical choice for this scale of data without significant infrastructure optimization. While training the model with limited samples (100 samples), it provided an accuracy of ~60%. 
