# rulebased.py

from dataclasses import dataclass
import re
import os
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Iterable, Set, Optional

from .configuration import LANG_CODES, STOPWORDS_PATH

# Load Stopwords
def _load_stopwords(lang: str, base_dir: str = STOPWORDS_PATH) -> Set[str]:
    file_path = os.path.join(base_dir, f"{lang}.txt")

    if not os.path.exists(file_path):
        return set()

    out: Set[str] = set()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.add(s.lower())
    return out

# Build language specific rules (script, stopwords, special characters)
def _build_lang_rules(langs: Iterable[str], base_dir: str = STOPWORDS_PATH) -> Dict[str, Dict]:
    meta: Dict[str, Dict] = {
        "en": {"script": "latin", "special_chars": set()},
        "de": {"script": "latin", "special_chars": {"ä", "ö", "ü", "ß"}},
        "es": {"script": "latin", "special_chars": {"ñ"}},
        "fr": {"script": "latin", "special_chars": {"é", "è", "ê", "à", "ç", "ù", "ô"}},
        "it": {"script": "latin", "special_chars": {"à", "è", "é", "ì", "ò", "ù"}},
        "pt": {"script": "latin", "special_chars": {"ã", "õ"}},
        "ko": {"script": "hangul", "special_chars": set()},
        "ta": {"script": "tamil", "special_chars": set()},
        "ru": {"script": "cyrillic", "special_chars": {"ъ", "ы", "э"}},
        "be": {"script": "cyrillic", "special_chars": {"ў", "і"}},
    }

    rules: Dict[str, Dict] = {}
    for lang in langs:
        m = meta.get(lang, {"script": "mixed", "special_chars": set()})
        rules[lang] = {
            "script": m["script"],
            "stopwords": _load_stopwords(lang, base_dir=base_dir),
            "special_chars": m["special_chars"],
        }
    return rules

# Set parameters with the best result
@dataclass
class RuleTaggerConfig:
    max_sent_per_lang: int = 50000
    top_k_n_grams: int = 40
    weights: str = "base"
    stopwords_dir: Optional[str] = None

# Tag Script Token
class RuleTagger:
    WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

    WEIGHT_PRESETS = {
        "base": dict(stopword=3.0, special=4.0, bi=1.0, tri=2.0, four=3.0)
    }

    SCRIPT_TOKEN = {
        "latin": "<SCRIPT=LATIN>",
        "cyrillic": "<SCRIPT=CYRILLIC>",
        "hangul": "<SCRIPT=HANGUL>",
        "tamil": "<SCRIPT=TAMIL>",
        "mixed": "<SCRIPT=MIXED>",
    }

    LANG_RULES: Dict[str, Dict] = {}

    def __init__(self, cfg: RuleTaggerConfig):
        self.cfg = cfg
        self.w = self.WEIGHT_PRESETS[cfg.weights]

        stop_dir = cfg.stopwords_dir or STOPWORDS_PATH
        self.LANG_RULES = _build_lang_rules(LANG_CODES, base_dir=stop_dir)

        self.ngrams = {lang: {"bi": set(), "tri": set(), "four": set()} for lang in LANG_CODES}
        self.fitted = False

    # Detect script (latin, cyrillic, hangul, tamil, or mixed)
    def detect_script(self, text: str) -> str:
        has_cyr = has_lat = has_hangul = has_tamil = False
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

        flags = [has_cyr, has_lat, has_hangul, has_tamil]
        if sum(flags) == 1:
            if has_cyr: return "cyrillic"
            if has_lat: return "latin"
            if has_hangul: return "hangul"
            if has_tamil: return "tamil"
        return "mixed"

    # Tokenize text into lowercase
    def tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in self.WORD_RE.findall(text)]

    # Extract character n-grams
    def char_ngrams(self, text: str, n: int) -> Counter:
        text = text.lower()
        chars = [c for c in text if c.isalpha()]
        grams = ["".join(chars[i:i+n]) for i in range(len(chars) - n + 1)]
        return Counter(grams)

    # Build top-k character n-grams per language
    def _build_top_ngrams(
        self,
        texts: Iterable[str],
        labels: Iterable[str],
        n: int,
        top_k: int,
        min_freq: int = 5,
    ) -> Dict[str, List[str]]:
        per_lang = defaultdict(Counter)
        for t, y in zip(texts, labels):
            per_lang[y].update(self.char_ngrams(t, n))
        out = {}
        for lang, c in per_lang.items():
            c = Counter({g: cnt for g, cnt in c.items() if cnt >= min_freq})
            out[lang] = [g for g, _ in c.most_common(top_k)]
        return out
    
    # Train language specific character n-grams
    def fit(self, train_texts: List[str], train_labels: List[str]) -> None:
        k = self.cfg.top_k_n_grams
        bi = self._build_top_ngrams(train_texts, train_labels, 2, k)
        tri = self._build_top_ngrams(train_texts, train_labels, 3, k)
        four = self._build_top_ngrams(train_texts, train_labels, 4, k)

        for lang in LANG_CODES:
            if lang in bi: self.ngrams[lang]["bi"] = set(bi[lang])
            if lang in tri: self.ngrams[lang]["tri"] = set(tri[lang])
            if lang in four: self.ngrams[lang]["four"] = set(four[lang])

        self.fitted = True

    # Scoring + Prediction
    def score_language(self, text: str, lang: str) -> float:
        rules = self.LANG_RULES[lang]
        toks = self.tokenize(text)

        bi_cnt = self.char_ngrams(text, 2)
        tri_cnt = self.char_ngrams(text, 3)
        four_cnt = self.char_ngrams(text, 4)

        s = 0.0

        for tok in toks:
            if tok in rules["stopwords"]:
                s += self.w["stopword"]

        for ch in text.lower():
            if ch in rules["special_chars"]:
                s += self.w["special"]

        for g in self.ngrams[lang]["bi"]:
            s += self.w["bi"] * bi_cnt.get(g, 0)
        for g in self.ngrams[lang]["tri"]:
            s += self.w["tri"] * tri_cnt.get(g, 0)
        for g in self.ngrams[lang]["four"]:
            s += self.w["four"] * four_cnt.get(g, 0)

        return s
    
    # Predict language based on rules and compute confidence margin
    def rule_predict(self, text: str) -> Tuple[str, float, float]:
        script = self.detect_script(text)
        candidates = [
            lang for lang in LANG_CODES
            if script == "mixed" or self.LANG_RULES[lang]["script"] == script
        ] or LANG_CODES[:]

        scores = {lang: self.score_language(text, lang) for lang in candidates}
        best_lang, best_score = max(scores.items(), key=lambda x: x[1])
        sorted_scores = sorted(scores.values(), reverse=True)
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = best_score - second
        return best_lang, best_score, margin

    # Confidence margin -> confidence level (LOW / MID / HIGH)
    def conf_bin(self, best_score: float, margin: float) -> str:
        if best_score <= 0:
            return "LOW"
        if margin >= 15:
            return "HIGH"
        if margin >= 5:
            return "MID"
        return "LOW"

    # Generate rule-based prefix tokens for model input
    def prefix(self, text: str) -> str:
        script_tok = self.SCRIPT_TOKEN.get(self.detect_script(text), "<SCRIPT=MIXED>")

        if not self.fitted:
            return f"{script_tok} <RB=en> <RB_CONF=LOW>"

        rb_lang, best_score, margin = self.rule_predict(text)
        conf = self.conf_bin(best_score, margin)
        return f"{script_tok} <RB={rb_lang}> <RB_CONF={conf}>"

    # Return all special tokens used by the tagger
    def special_tokens(self) -> List[str]:
        script_tokens = list(self.SCRIPT_TOKEN.values())
        rb_lang_tokens = [f"<RB={l}>" for l in LANG_CODES]
        rb_conf_tokens = ["<RB_CONF=LOW>", "<RB_CONF=MID>", "<RB_CONF=HIGH>"]
        return script_tokens + rb_lang_tokens + rb_conf_tokens