import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report, accuracy_score


# Data Loading and Parsing 
def parse_conllu_data(file_path, language_code):
    """Parses a custom CoNLL-U file and extracts sentences and their labels."""
    data = []
    current_sentence = ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# text = '):
                    if current_sentence:
                        data.append({'text': current_sentence, 'label': language_code})
                    current_sentence = line.split('=', 1)[1].strip()
                elif not line and current_sentence:
                    data.append({'text': current_sentence, 'label': language_code})
                    current_sentence = ""
            if current_sentence:
                 data.append({'text': current_sentence, 'label': language_code})
                 
    except FileNotFoundError:
        return pd.DataFrame() 
        
    return pd.DataFrame(data).drop_duplicates()

data_dir = 'data'
languages = ['be', 'de', 'en', 'es', 'fr', 'it', 'ko', 'pt', 'ru', 'ta']

all_data = []
print("--- Parsing Data ---")
for lang_code in languages:
    filename = f'output_{lang_code}.conllu'
    file_path = os.path.join(data_dir, filename)
    df = parse_conllu_data(file_path, lang_code)
    if not df.empty:
        print(f"Parsed file: {filename} with {len(df)} sentences.")
        all_data.append(df)
    
df_combined = pd.concat(all_data, ignore_index=True)

# Data Preparation 
N_SAMPLES = 6669377
total_samples = len(df_combined)

if total_samples == 0:
    print("\nError: No data was loaded.")
else:
    print(f"\nTotal sentences loaded: {total_samples}")

    if total_samples >= N_SAMPLES:
        df_sampled = df_combined.sample(n=N_SAMPLES, random_state=42)
        print(f"Successfully sampled {N_SAMPLES} sentences.")
    else:
        df_sampled = df_combined
        print(f"Warning: Only {total_samples} available, using all data instead of {N_SAMPLES}.")

    le = LabelEncoder()
    df_sampled['label_id'] = le.fit_transform(df_sampled['label'])
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Label Mapping:", label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        df_sampled['text'], 
        df_sampled['label_id'], 
        test_size=0.2, 
        random_state=42,
        stratify=df_sampled['label_id']
    )

    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"\nTraining set size: {X_train_vec.shape}")
    print(f"Testing set size: {X_test_vec.shape}")

    # Model Training and Evaluation 
    print("\nTraining the Multinomial Naive Bayes Model...")

    model = MultinomialNB() 
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    print("\nClassification Report (Language ID -> Language Code):")
    class_names = le.classes_
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
