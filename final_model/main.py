# main.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from .load_data import load_wikipedia_data, load_twitter_data
from .finetuned_xlmroberta import FineTunedRobertaModel
from .configuration import TEST_RATIO, RANDOM_SEED
from .rulebased import RuleTagger, RuleTaggerConfig

import matplotlib.pyplot as plt

def main():
    # Load Dataset
    wiki_texts, wiki_labels = load_wikipedia_data(max_per_lang=50000)

    # Split train/test
    wiki_train_texts, wiki_test_texts, wiki_train_labels, wiki_test_labels = train_test_split(
        wiki_texts, wiki_labels,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=wiki_labels
    )

    print(f"\nWikipedia Train: {len(wiki_train_texts)}")
    print(f"Wikipedia Test: {len(wiki_test_texts)}")

    twitter_texts, twitter_labels = load_twitter_data()

    # Train Fine-Tuned XLM-RoBERTa Model
    tagger = RuleTagger(RuleTaggerConfig(top_k_n_grams=40, max_sent_per_lang=50000, weights="base"))
    finetune_roberta_model = FineTunedRobertaModel(tagger)
    finetune_roberta_model.train(wiki_train_texts, wiki_train_labels, max_samples=50000)

    # Evaluation on Wikipedia
    print("\n" + "="*50)
    print("[EVALUATION ON WIKIPEDIA]")
    print("="*50)

    wiki_preds = finetune_roberta_model.predict(wiki_test_texts)
    wiki_acc = accuracy_score(wiki_test_labels, wiki_preds)
    print(f"\nAccuracy: {wiki_acc:.4f} ({wiki_acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(wiki_test_labels, wiki_preds))
    
    # Confusion Matrix
    labels = sorted(list(set(wiki_train_labels) | set(wiki_test_labels) | set(wiki_labels)))

    wiki_cm = confusion_matrix(wiki_test_labels, wiki_preds, labels=labels)
    print("\nConfusion Matrix:")
    print(wiki_cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=wiki_cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Wikipedia")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Evaluation on Twitter
    if twitter_texts:
        print("\n" + "="*50)
        print("[EVALUATION ON TWITTER]")
        print("="*50)

        twitter_preds = finetune_roberta_model.predict(twitter_texts)
        twitter_acc = accuracy_score(twitter_labels, twitter_preds)
        print(f"\nAccuracy: {twitter_acc:.4f} ({twitter_acc*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(twitter_labels, twitter_preds))
        
        # Confusion Matrix
        twitter_cm = confusion_matrix(twitter_labels, twitter_preds, labels=labels)
        print("\nConfusion Matrix:")
        print(twitter_cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=twitter_cm, display_labels=labels)
        disp.plot(cmap="Oranges", values_format="d")
        plt.title("Confusion Matrix - Twitter")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"XLM-RoBERTa on Wikipedia: {wiki_acc*100:.2f}%")
    if twitter_texts:
        print(f"XLM-RoBERTa on Twitter: {twitter_acc*100:.2f}%")
        print(f"Drop: {(wiki_acc - twitter_acc)*100:.2f}%")


if __name__ == "__main__":
    main()
