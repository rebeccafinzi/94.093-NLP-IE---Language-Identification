import re
import unicodedata
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Iterable

# ----------------- 0. Языки, с которыми работаем -----------------

LANG_CODES = ["de", "en", "es", "fr", "it", "ko", "pt", "ta", "be", "ru"]

# ----------------- 1. Базовые правила по языкам (ручные) -----------------

LANG_RULES: Dict[str, Dict] = {
    "en": {
        "name": "English",
        "script": "latin",
        "stopwords": {
            "the", "and", "of", "to", "in", "is", "for",
            "on", "with", "that", "this", "it", "as", "at"
        },
        "special_chars": set(),
        "bigrams": set(),
        "trigrams": set(),
    },
    "de": {
        "name": "German",
        "script": "latin",
        "stopwords": {
            "der", "die", "das", "und", "nicht", "ist",
            "ein", "eine", "im", "mit", "für", "auf"
        },
        "special_chars": {"ä", "ö", "ü", "ß"},
        "bigrams": set(),
        "trigrams": set(),
    },
    "es": {
        "name": "Spanish",
        "script": "latin",
        "stopwords": {
            "el", "la", "los", "las",
        "del", "al",
        "que", "porque", "como",
        "una", "uno",
        "sí", "pero"
        },
        "special_chars": {"ñ"},
        "bigrams": set(),
        "trigrams": set(),
    },
    "fr": {
        "name": "French",
        "script": "latin",
        "stopwords": {
            "le", "la", "les",
            "de", "des",
            "un", "une",
            "et", "est",
            "pour", "sur",
            "au", "aux",
            "pas", "ces", "cette"
        },
        "special_chars": {"é", "è", "ê", "à", "ç", "ù", "ô"},
        "bigrams": set(),
        "trigrams": set(),
    },
    "it": {
        "name": "Italian",
        "script": "latin",
        "stopwords": {
            "il", "lo", "la", "i", "gli", "le",
            "di", "che", "per", "con", "nel",
            "della", "dello"
        },
        "special_chars": {"à", "è", "é", "ì", "ò", "ù"},
        "bigrams": set(),
        "trigrams": set(),
    },
    "pt": {
        "name": "Portuguese",
        "script": "latin",
        "stopwords": {
            "o", "a", "os", "as",
            "do", "da", "dos", "das",
            "que", "em", "para", "por",
            "não", "também", "nos", "nas"
        },
        "special_chars": {"ã", "õ"},
        "bigrams": set(),
        "trigrams": set(),
    },
    "ko": {
        "name": "Korean",
        "script": "hangul",
        "stopwords": {
            "이", "그", "저", "그리고", "하지만", "에서", "에게", "이다"
        },
        "special_chars": set(),  # сам хангыль уже отличим
        "bigrams": set(),
        "trigrams": set(),
    },
    "ta": {
        "name": "Tamil",
        "script": "tamil",
        "stopwords": {
            "ஒரு", "இந்த", "அது", "இல்", "மற்றும்", "என்று"
        },
        "special_chars": set(),
        "bigrams": set(),
        "trigrams": set(),
    },
    "ru": {
        "name": "Russian",
        "script": "cyrillic",
        "stopwords": {
            "что", "это", "чтобы", "которые", "который",
            "или", "если", "потому", "также",
            "будет", "были", "может", "нужно"
        },
        "special_chars": {"ъ", "ы", "э"},
        "bigrams": set(),
        "trigrams": set(),
    },
    "be": {
        "name": "Belarusian",
        "script": "cyrillic",
        "stopwords": {
            "і", "што", "ён", "яна", 
            "як", "з", "па", "гэта", "калі", "каб", "яшчэ", "таму",
            "ён", "яны", "які", "якая", "якое", "якія", "нібыта"
        },
        "special_chars": {"ў", "і"},
        "bigrams": set(),
        "trigrams": set(),
    },
}

# ----------------- 2. Скрипт текста -----------------

def detect_script(text: str) -> str:
    has_cyr = False
    has_lat = False
    has_hangul = False
    has_tamil = False

    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name:
            has_cyr = True
        elif "LATIN" in name:
            has_lat = True
        elif "HANGUL" in name:
            has_hangul = True
        elif "TAMIL" in name:
            has_tamil = True

    # приоритет: если явно один скрипт
    flags = [has_cyr, has_lat, has_hangul, has_tamil]
    if sum(flags) == 1:
        if has_cyr:
            return "cyrillic"
        if has_lat:
            return "latin"
        if has_hangul:
            return "hangul"
        if has_tamil:
            return "tamil"

    return "mixed"


# ----------------- 3. Токенизация и n-граммы -----------------

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def char_ngrams(text: str, n: int) -> Counter:
    text = text.lower()
    chars = [c for c in text if c.isalpha()]
    grams = ["".join(chars[i:i + n]) for i in range(len(chars) - n + 1)]
    return Counter(grams)


# ----------------- 4. Чтение .conllu -----------------

def load_conllu_sentences(path: str) -> List[str]:
    """
    Вытаскиваем все строки '# text = ...' как отдельные предложения.
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# text ="):
                sent = line[len("# text ="):].strip()
                if sent:
                    sentences.append(sent)
    return sentences


def load_corpus_from_dir(data_dir: str,
                         max_sent_per_lang: int = 50000) -> Tuple[List[str], List[str]]:
    """
    Читает все output_XX.conllu из data_dir.
    Возвращает:
        texts: список предложений
        labels: соответствующие языковые коды
    """
    texts, labels = [], []
    for lang in LANG_CODES:
        path = f"{data_dir}/output_{lang}.conllu"
        try:
            sents = load_conllu_sentences(path)
        except FileNotFoundError:
            print(f"WARNING: file not found for language {lang}: {path}")
            continue

        if max_sent_per_lang is not None:
            sents = sents[:max_sent_per_lang]

        texts.extend(sents)
        labels.extend([lang] * len(sents))
    return texts, labels


# ----------------- 5. Построение n-грамм на корпусе -----------------

def build_char_ngrams_from_corpus(
    texts: Iterable[str],
    labels: Iterable[str],
    n: int,
    top_k: int = 30,
    min_freq: int = 5,
) -> Dict[str, List[str]]:
    per_lang_counts: Dict[str, Counter] = defaultdict(Counter)

    for text, lang in zip(texts, labels):
        grams = char_ngrams(text, n)
        per_lang_counts[lang].update(grams)

    result: Dict[str, List[str]] = {}
    for lang, counter in per_lang_counts.items():
        filtered = Counter({g: c for g, c in counter.items() if c >= min_freq})
        result[lang] = [g for g, _ in filtered.most_common(top_k)]
    return result


def init_ngram_rules(
    texts: Iterable[str],
    labels: Iterable[str],
    bigram_top_k: int = 30,
    trigram_top_k: int = 30,
) -> None:
    """
    «Обучаем» биграммы и триграммы из train-корпуса и
    записываем внутрь LANG_RULES.
    """
    bigrams = build_char_ngrams_from_corpus(texts, labels, n=2, top_k=bigram_top_k)
    trigrams = build_char_ngrams_from_corpus(texts, labels, n=3, top_k=trigram_top_k)

    for lang, rules in LANG_RULES.items():
        if lang in bigrams:
            rules["bigrams"] = set(bigrams[lang])
        if lang in trigrams:
            rules["trigrams"] = set(trigrams[lang])


# ----------------- 6. Scoring для одного языка -----------------

def score_language(text: str, lang_code: str) -> float:
    rules = LANG_RULES[lang_code]
    tokens = tokenize(text)
    bigram_counts = char_ngrams(text, 2)
    trigram_counts = char_ngrams(text, 3)

    score = 0.0

    # стоп-слова
    sw = rules.get("stopwords", set())
    for tok in tokens:
        if tok in sw:
            score += 1.5

    # спец-символы
    specials = rules.get("special_chars", set())
    for ch in text.lower():
        if ch in specials:
            score += 4.0

    # биграммы
    for gram in rules.get("bigrams", set()):
        score += 1.0 * bigram_counts.get(gram, 0)

    # триграммы
    for gram in rules.get("trigrams", set()):
        score += 2.0 * trigram_counts.get(gram, 0)

    return score


# ----------------- 7. Предсказание языка -----------------

def predict_language(text: str) -> str:
    script = detect_script(text)

    candidates = []
    for code, cfg in LANG_RULES.items():
        if script == "mixed":
            candidates.append(code)
        elif cfg["script"] == script:
            candidates.append(code)

    if not candidates:
        candidates = list(LANG_RULES.keys())

    scores = {code: score_language(text, code) for code in candidates}
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score == 0:
        return "unknown"
    return best_lang


# ----------------- 8. Оценка -----------------

def evaluate(texts: List[str], labels: List[str]) -> Tuple[float, Counter]:
    assert len(texts) == len(labels)
    correct = 0
    conf = Counter()
    for t, gold in zip(texts, labels):
        pred = predict_language(t)
        if pred == gold:
            correct += 1
        conf[(gold, pred)] += 1
    acc = correct / len(texts) if texts else 0.0
    return acc, conf


# ----------------- 9. Простейший end-to-end запуск -----------------

def train_test_split(texts: List[str], labels: List[str], test_ratio: float = 0.2, seed: int = 42):
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * (1 - test_ratio))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    return train_texts, train_labels, test_texts, test_labels


def demo_example():
    examples = [
        "This is a simple English sentence.",
        "Das ist ein deutscher Beispielsatz.",
        "Ceci est une phrase française.",
        "Este es un ejemplo en español.",
        "Questo è un esempio italiano.",
        "Это простой русский пример.",
        "я зрабіў прыкладны беларускі сказ",
    ]
    for s in examples:
        print(predict_language(s), "->", s)

# ---------------- debug -----------------

def debug_scores(text: str):
    print("TEXT:", text)
    print("SCRIPT:", detect_script(text))

    bigram_counts = char_ngrams(text, 2)
    trigram_counts = char_ngrams(text, 3)

    for lang, rules in LANG_RULES.items():
        s = score_language(text, lang)
        print(f"\nLanguage {lang} score = {s}")
        print("  stopwords:",
              [tok for tok in tokenize(text) if tok in rules["stopwords"]])
        print("  special chars:",
              [ch for ch in text.lower() if ch in rules["special_chars"]])
        print("  bigrams:",
              [g for g in rules["bigrams"] if bigram_counts.get(g, 0) > 0][:20])
        print("  trigrams:",
              [g for g in rules["trigrams"] if trigram_counts.get(g, 0) > 0][:20])

if __name__ == "__main__":
    DATA_DIR = "preprocessing/data"  # папка, где лежат output_*.conllu

    # 1) грузим все предложения из conllu
    texts, labels = load_corpus_from_dir(DATA_DIR, max_sent_per_lang=50000)
    print(f"Total sentences: {len(texts)}")

    # 2) делим на train/test
    train_texts, train_labels, test_texts, test_labels = train_test_split(texts, labels, test_ratio=0.2)

    # 3) обучаем n-граммные правила на train
    init_ngram_rules(train_texts, train_labels, bigram_top_k=30, trigram_top_k=30)

    debug_scores("я зрабіў прыкладны беларускі сказ")

    print("\nSample bigrams/trigrams per language:")
for lang in ["en", "de", "fr", "es", "ru", "be"]:
    rules = LANG_RULES[lang]
    print(f"\n{lang} bigrams:", list(sorted(rules["bigrams"]))[:20])
    print(f"{lang} trigrams:", list(sorted(rules["trigrams"]))[:20])

    # 4) оцениваем на test
    acc, conf = evaluate(test_texts, test_labels)
    print(f"Rule-based accuracy: {acc:.4f}")
    print("Some confusions:")
    for (gold, pred), cnt in conf.most_common(20):
        print(f"{gold} -> {pred}: {cnt}")

    # 5) маленькая демо-проверка на руками заданных примерах
    print("\nDemo examples:")
    demo_example()
