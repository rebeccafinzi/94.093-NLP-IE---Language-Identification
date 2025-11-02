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
 
4. **Stopword removal** – Removal of very common words such as "the", "and", "di", "e".

5. **Stemming / Lemmatization** – Reduce tokens to their root or base forms.

6. **Saving in CoNLL format** – Finally, the cleaned and tokenized text was saved in CoNLL-U format, suitable for NLP model training and evaluation.  
   


