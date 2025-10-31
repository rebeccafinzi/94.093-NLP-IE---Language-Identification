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
- Malayalam  
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

1. **Removal of punctuation and symbols** – All non-alphanumeric characters, special symbols, and unnecessary whitespace were removed to reduce noise.
   
2. **Lowercasing** – Converted all text to lowercase to reduce variation caused by case differences.

3. **Tokenization** – Split text into individual tokens (words or meaningful sub-units) while respecting language-specific rules.
 
4. **Stopword removal** – Removal of very common words such as "the", "and", "di", "e".

5. **Stemming / Lemmatization** – Reduce tokens to their root or base forms.

6. **Saving in CoNLL format** – Finally, save the cleaned and preprocessed data in CoNLL format for use in NLP models.
   


