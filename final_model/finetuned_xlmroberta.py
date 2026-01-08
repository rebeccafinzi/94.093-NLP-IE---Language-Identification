# finetuned_xlmroberta.py

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from .configuration import LANG_CODES
from .rulebased import RuleTagger

# Rule-Augmented XLM-RoBERTa
class FineTunedRobertaModel:
    # Initialize model configuration and label mappings
    def __init__(self, tagger: RuleTagger):
        self.tagger = tagger

        self.model_name = "xlm-roberta-base"
        self.max_length = 128
        self.batch_size = 32
        self.num_epochs = 3
        self.learning_rate = 2e-5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {self.device}")

        self.label2id = {lang: i for i, lang in enumerate(LANG_CODES)}
        self.id2label = {i: lang for i, lang in enumerate(LANG_CODES)}

        self.model = None
        self.tokenizer = None

    # Load tokenizer and XLM-RoBERTa model
    # Add rule-based special tokens
    def _load_model(self):
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Add tagger's tokens
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": self.tagger.special_tokens()
        })

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(LANG_CODES),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        print("Model loaded")

    # Prepend rule-based prefix tokens to input texts
    def _apply_prefixes(self, texts):
        return [f"{self.tagger.prefix(t)} {t}" for t in texts]

    # Fine-tune XLM-RoBERTa on rule-augmented inputs
    def train(self, train_texts, train_labels, max_samples=50_000):
        if self.model is None:
            self._load_model()

        # Cap samples like before
        if len(train_texts) > max_samples:
            idx = np.random.choice(len(train_texts), max_samples, replace=False)
            train_texts = [train_texts[i] for i in idx]
            train_labels = [train_labels[i] for i in idx]

        # Fit rule tagger on the training split
        self.tagger.fit(train_texts, train_labels)
        
        # Augment inputs with rule-based prefix tokens
        tagged_texts = self._apply_prefixes(train_texts)

        # Tokenize rule-augmented inputs
        enc = self.tokenizer(
            tagged_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        y = torch.tensor([self.label2id[l] for l in train_labels], dtype=torch.long)

        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # AdamW optimizer for stable transformer finetuning
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss, correct, total = 0.0, 0, 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for input_ids, attention_mask, yb in pbar:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=yb)
                loss = out.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(out.logits, dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

            print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}, acc={correct/total:.4f}")

        print("Completed")

    # Predict languages
    def predict(self, texts):
        if self.model is None or self.tokenizer is None:
            self._load_model()

        self.model.eval()
        tagged_texts = self._apply_prefixes(texts)

        preds_all = []
        for i in tqdm(range(0, len(tagged_texts), self.batch_size), desc="Predict"):
            batch = tagged_texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model(**inputs)
                preds = torch.argmax(out.logits, dim=-1).cpu().tolist()

            preds_all.extend([self.id2label[p] for p in preds])

        return preds_all