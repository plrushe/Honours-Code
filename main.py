import flet as ft
import json
import calendar
from pathlib import Path
from datetime import date, datetime, timedelta

# Optional transformers support for deep learning model
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    torch = None
    HAS_TRANSFORMERS = False

# --- ML imports ---
import joblib

# --- TextCleaner dependencies (must exist for joblib.load) ---
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


DATA_DIR = Path("data")
ENTRIES_BY_DATE_FILE = DATA_DIR / "entries_by_date.json"

MODELS_DIR = Path("models")

# Classic ML model
MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH = MODELS_DIR / "meta.joblib"

# Deep model folder
DL_MODEL_DIR = (MODELS_DIR / "distilbert_depression_classifier").resolve()


def today_iso():
    return date.today().isoformat()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


# -------------------------
# Storage: dict keyed by date
# -------------------------
def load_entries_by_date() -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    if not ENTRIES_BY_DATE_FILE.exists():
        return {}
    try:
        data = json.loads(ENTRIES_BY_DATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_entries_by_date(entries: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    tmp = ENTRIES_BY_DATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ENTRIES_BY_DATE_FILE)


def upsert_entry_by_date(day: str, entry_obj: dict) -> None:
    entries = load_entries_by_date()
    entries[day] = entry_obj
    save_entries_by_date(entries)


def delete_entry_by_date(day: str) -> None:
    entries = load_entries_by_date()
    if day in entries:
        del entries[day]
        save_entries_by_date(entries)


# -------------------------
# IMPORTANT: TextCleaner (must exist for joblib.load)
# -------------------------
def ensure_nltk():
    for path, pkg in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


ensure_nltk()

STOPWORDS = set(stopwords.words("english")) - {"not"}
LEMM = WordNetLemmatizer()
URL_RE = re.compile(r"http\S+|www\.\S+")


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for t in X:
            t = str(t).lower()
            t = URL_RE.sub(" ", t)
            t = t.translate(str.maketrans("", "", string.punctuation))
            tokens = nltk.word_tokenize(t)

            out = []
            for w in tokens:
                if w in STOPWORDS:
                    continue
                if not w.isalpha():
                    continue
                out.append(LEMM.lemmatize(w))

            cleaned.append(" ".join(out))
        return np.array(cleaned, dtype=object)


# -------------------------
# Shared language helpers
# -------------------------
def normalize_label(lbl: str) -> str:
    s = str(lbl or "").strip().lower()
    if s in ("no depression", "no_depression", "nodepression", "positive"):
        return "positive"
    if s in ("depression", "negative"):
        return "negative"
    if s in ("error", "unknown", "unavailable"):
        return "unknown"
    return s or "unknown"


def label_to_sentiment_class(lbl: str) -> str:
    nl = normalize_label(lbl)
    if nl == "positive":
        return "pos"
    if nl == "negative":
        return "neg"
    return "unk"


def get_p_negative(sentiment: dict):
    if not isinstance(sentiment, dict):
        return None
    if "p_negative" in sentiment:
        return sentiment.get("p_negative")
    if "p_depression" in sentiment:
        return sentiment.get("p_depression")
    return None


def tone_from_probability(p_neg):
    if p_neg is None:
        return "Mixed tone"
    if p_neg >= 0.60:
        return "Negative tone"
    if p_neg <= 0.40:
        return "Positive tone"
    return "Mixed tone"

def tone_from_result(result: dict) -> str:
    if not result or not result.get("available", True):
        return "Unavailable"

    lbl = normalize_label(result.get("label"))
    if lbl == "negative":
        return "Negative tone"
    if lbl == "positive":
        return "Positive tone"
    return "Mixed tone"

def signal_strength(p_neg):
    if p_neg is None:
        return "Unknown"
    if p_neg >= 0.75 or p_neg <= 0.25:
        return "High"
    if p_neg >= 0.55 or p_neg <= 0.35:
        return "Moderate"
    return "Low"


def average_tone_text(avg_p_neg):
    if avg_p_neg is None:
        return "Not enough data yet"
    if avg_p_neg >= 0.65:
        return "Mostly negative tone"
    if avg_p_neg >= 0.45:
        return "Slightly negative tone"
    if avg_p_neg <= 0.35:
        return "Mostly positive tone"
    return "Fairly balanced"


def probability_to_confidence(p_neg):
    if p_neg is None:
        return None
    return abs(float(p_neg) - 0.5)


def model_confidence_text(p_neg):
    c = probability_to_confidence(p_neg)
    if c is None:
        return "Unavailable"
    if c >= 0.25:
        return "High confidence"
    if c >= 0.12:
        return "Moderate confidence"
    return "Low confidence"


def friendly_date_text(day_iso: str) -> str:
    try:
        return datetime.strptime(day_iso, "%Y-%m-%d").strftime("%d %b %Y")
    except Exception:
        return day_iso


def compare_model_outputs(ml_result: dict, dl_result: dict) -> str:
    if not ml_result:
        return "No model result available."
    if not dl_result or not dl_result.get("available", False):
        return "Deep model is not available yet."

    ml_lbl = normalize_label(ml_result.get("label"))
    dl_lbl = normalize_label(dl_result.get("label"))

    if ml_lbl == dl_lbl and ml_lbl == "positive":
        return "Both models suggest a more positive or steady tone in this entry."
    if ml_lbl == dl_lbl and ml_lbl == "negative":
        return "Both models suggest a more difficult or negative tone in this entry."
    return "The models interpret this entry differently."


def model_agreement_status(ml_result: dict, dl_result: dict) -> str:
    if not ml_result or not dl_result or not dl_result.get("available", False):
        return "Not enough data for comparison."
    return "Agreement" if normalize_label(ml_result.get("label")) == normalize_label(dl_result.get("label")) else "Disagreement"


def confidence_comparison_text(ml_result: dict, dl_result: dict) -> str:
    if not ml_result:
        return "Confidence comparison unavailable."
    if not dl_result or not dl_result.get("available", False):
        return "Confidence comparison unavailable because the deep model is not available."

    ml_p = get_p_negative(ml_result)
    dl_p = get_p_negative(dl_result)

    ml_c = probability_to_confidence(ml_p)
    dl_c = probability_to_confidence(dl_p)

    if ml_c is None or dl_c is None:
        return "Confidence comparison unavailable."

    if abs(ml_c - dl_c) < 0.03:
        return "Both models show a similar level of confidence."
    if ml_c > dl_c:
        return "The classic model is more confident for this entry."
    return "The deep model is more confident for this entry."


def reflection_prompt_for_entry(ml_result: dict, dl_result: dict) -> str:
    ml_p = get_p_negative(ml_result)
    dl_p = get_p_negative(dl_result)
    avg = None

    vals = [v for v in [ml_p, dl_p] if v is not None]
    if vals:
        avg = sum(vals) / len(vals)

    status = model_agreement_status(ml_result, dl_result)

    if avg is not None and avg >= 0.68:
        return "Reflection prompt: What felt most difficult today, and was there even one small moment that made things feel lighter?"
    if status == "Disagreement":
        return "Reflection prompt: The two models interpreted this entry differently. Was your day mixed, with both difficult and steady moments?"
    if avg is not None and avg <= 0.32:
        return "Reflection prompt: What helped today feel steadier or more positive, and is that something you could repeat tomorrow?"
    return "Reflection prompt: What stood out most about today, and what seems to have influenced your mood the most?"


def consecutive_negative_days(entries: dict, lookback_days: int = 30) -> int:
    count = 0
    current = date.today()

    for offset in range(lookback_days):
        day = (current - timedelta(days=offset)).isoformat()
        obj = entries.get(day)
        if not obj:
            break

        ml = obj.get("sentiment_ml")
        if ml is None and isinstance(obj.get("sentiment"), dict):
            ml = obj.get("sentiment")

        if not ml:
            break

        if label_to_sentiment_class(ml.get("label", "")) == "neg":
            count += 1
        else:
            break

    return count


def support_prompt_text(entries: dict) -> str:
    streak = consecutive_negative_days(entries, lookback_days=14)
    if streak >= 3:
        return (
            "Support note: Your recent entries have shown a more difficult tone for several days in a row. "
            "This app is only a reflection tool and not a diagnosis. If things feel persistently hard, "
            "consider reaching out to someone you trust or a qualified professional."
        )
    return ""


def current_entry_streak(entries: dict) -> int:
    if not entries:
        return 0

    count = 0
    current = date.today()

    while True:
        day = current.isoformat()
        if day in entries:
            count += 1
            current -= timedelta(days=1)
        else:
            break
    return count


def longest_entry_streak(entries: dict) -> int:
    if not entries:
        return 0

    days = sorted(entries.keys())
    best = 1
    current = 1

    for i in range(1, len(days)):
        try:
            d_prev = datetime.strptime(days[i - 1], "%Y-%m-%d").date()
            d_now = datetime.strptime(days[i], "%Y-%m-%d").date()
            if d_now == d_prev + timedelta(days=1):
                current += 1
                best = max(best, current)
            else:
                current = 1
        except Exception:
            current = 1

    return best


def get_period_stats(entries: dict, days_back: int = 7) -> dict:
    end_day = date.today()
    start_day = end_day - timedelta(days=days_back - 1)
    keys = [(start_day + timedelta(days=i)).isoformat() for i in range(days_back)]

    present = 0
    pos = neg = unk = 0
    pvals = []
    agreements = 0
    disagreements = 0

    for k in keys:
        obj = entries.get(k)
        if not obj:
            continue

        ml = obj.get("sentiment_ml")
        if ml is None and isinstance(obj.get("sentiment"), dict):
            ml = obj.get("sentiment")

        dl = obj.get("sentiment_dl")

        if ml:
            present += 1
            cls = label_to_sentiment_class(ml.get("label", ""))
            if cls == "pos":
                pos += 1
            elif cls == "neg":
                neg += 1
            else:
                unk += 1

            p = get_p_negative(ml)
            if p is not None:
                try:
                    pvals.append(float(p))
                except Exception:
                    pass

        if ml and dl and dl.get("available", False):
            if normalize_label(ml.get("label")) == normalize_label(dl.get("label")):
                agreements += 1
            else:
                disagreements += 1

    avg = (sum(pvals) / len(pvals)) if pvals else None

    return {
        "days_back": days_back,
        "present": present,
        "missing": days_back - present,
        "pos": pos,
        "neg": neg,
        "unk": unk,
        "avg_p": avg,
        "agreements": agreements,
        "disagreements": disagreements,
    }


# -------------------------
# Classic ML model
# -------------------------
_ml_model = None
_ml_meta = None


def load_ml_model_and_meta():
    global _ml_model, _ml_meta
    if _ml_model is not None and _ml_meta is not None:
        return _ml_model, _ml_meta

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH.resolve()}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing meta file: {META_PATH.resolve()}")

    _ml_model = joblib.load(MODEL_PATH)
    _ml_meta = joblib.load(META_PATH)

    if not hasattr(_ml_model, "predict_proba"):
        raise TypeError(
            "Loaded classic model does not support predict_proba(). "
            "Make sure you exported the calibrated pipeline."
        )

    return _ml_model, _ml_meta


def predict_with_classic_model(text: str) -> dict:
    model, meta = load_ml_model_and_meta()

    # Use a stricter threshold for journal-tone use
    threshold = 0.55

    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)

    label_mapping = meta.get("label_mapping", {})
    negative_label_value = label_mapping.get("depression", 1)

    negative_idx = None
    for i, cls in enumerate(classes):
        if int(cls) == int(negative_label_value):
            negative_idx = i
            break

    if negative_idx is None:
        raise ValueError(f"Could not determine negative class from model.classes_: {classes}")

    p_neg = float(proba[negative_idx])

    pred = 1 if p_neg >= threshold else 0
    label = "negative" if pred == 1 else "positive"

    feedback = (
        "This model suggests the writing leans toward a more difficult emotional tone."
        if label == "negative"
        else "This model suggests the writing leans toward a more positive or steady emotional tone."
    )

    return {
        "label": label,
        "predicted_class": pred,
        "p_negative": p_neg,
        "threshold": threshold,
        "feedback": feedback,
        "model": "classic-ml",
        "available": True,
    }


# -------------------------
# Deep learning model
# -------------------------
_dl_tokenizer = None
_dl_model = None
_dl_load_attempted = False


def load_deep_model():
    global _dl_tokenizer, _dl_model, _dl_load_attempted

    if _dl_tokenizer is not None and _dl_model is not None:
        return _dl_tokenizer, _dl_model

    if _dl_load_attempted:
        return None, None

    _dl_load_attempted = True

    if not HAS_TRANSFORMERS:
        print("Transformers not installed.")
        return None, None

    if not DL_MODEL_DIR.exists():
        print("Deep model folder not found:", DL_MODEL_DIR)
        return None, None

    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    for name in required:
        p = DL_MODEL_DIR / name
        if not p.exists():
            print(f"Missing deep model file: {p}")
            return None, None

    try:
        _dl_tokenizer = AutoTokenizer.from_pretrained(
            str(DL_MODEL_DIR),
            local_files_only=True,
        )
        _dl_model = AutoModelForSequenceClassification.from_pretrained(
            str(DL_MODEL_DIR),
            local_files_only=True,
        )
        _dl_model.eval()
        print("Deep model loaded successfully.")
        return _dl_tokenizer, _dl_model
    except Exception as ex:
        print("Deep model load error:", ex)
        return None, None


def predict_with_deep_model(text: str) -> dict:
    tokenizer, model = load_deep_model()
    if tokenizer is None or model is None:
        return {
            "label": "unknown",
            "predicted_class": None,
            "p_negative": None,
            "threshold": None,
            "feedback": "Deep model unavailable.",
            "model": "deep-learning",
            "available": False,
        }

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        if len(probs) < 2:
            return {
                "label": "unknown",
                "predicted_class": None,
                "p_negative": None,
                "threshold": None,
                "feedback": "Deep model output shape was unexpected.",
                "model": "deep-learning",
                "available": False,
            }

        p_negative = float(probs[1])
        predicted_class = int(np.argmax(probs))
        label = "negative" if predicted_class == 1 else "positive"

        feedback = (
            "This model suggests the writing leans toward a more difficult emotional tone."
            if label == "negative"
            else "This model suggests the writing leans toward a more positive or steady emotional tone."
        )

        return {
            "label": label,
            "predicted_class": predicted_class,
            "p_negative": p_negative,
            "threshold": 0.5,
            "feedback": feedback,
            "model": "deep-learning",
            "available": True,
        }

    except Exception as ex:
        return {
            "label": "unknown",
            "predicted_class": None,
            "p_negative": None,
            "threshold": None,
            "feedback": f"Deep model error: {ex}",
            "model": "deep-learning",
            "available": False,
        }


def main(page: ft.Page):
    page.title = "Sentiment Journal"
    page.window_width = 430
    page.window_height = 760
    page.padding = 16
    page.scroll = ft.ScrollMode.AUTO
    page.theme_mode = ft.ThemeMode.DARK

    def set_route(route: str):
        page.route = route
        render()

    def refresh_and_render():
        render()

    def render():
        page.controls.clear()

        if page.route == "/":
            page.controls.append(home_screen())
        elif page.route == "/journal":
            page.controls.append(journal_screen())
        elif page.route == "/diary":
            page.controls.append(diary_screen())
        elif page.route == "/analysis":
            page.controls.append(analysis_screen())
        elif page.route == "/settings":
            page.controls.append(settings_screen())
        else:
            page.controls.append(not_found_screen())

        page.update()

    def top_bar(title: str):
        return ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.TextButton(content=ft.Text("Back"), on_click=lambda _: set_route("/")),
                ft.Text(title, size=18, weight=ft.FontWeight.BOLD),
                ft.Container(width=60),
            ],
        )

    def card(child, expand=False, bordered=True, bgcolor=None, padding=14):
        return ft.Container(
            expand=expand,
            padding=padding,
            border_radius=16,
            bgcolor=bgcolor,
            border=ft.Border.all(1, ft.Colors.with_opacity(0.12, ft.Colors.WHITE)) if bordered else None,
            content=child,
        )

    def model_result_block(title: str, result: dict):
        available = result.get("available", True)
        p_neg = get_p_negative(result)

        tone = tone_from_result(result)
        strength = signal_strength(p_neg) if available else "Unavailable"
        confidence = model_confidence_text(p_neg) if available else "Unavailable"

        badge_bg = ft.Colors.with_opacity(0.10, ft.Colors.WHITE)

        return ft.Container(
            padding=12,
            border_radius=14,
            border=ft.Border.all(1, ft.Colors.with_opacity(0.14, ft.Colors.WHITE)),
            content=ft.Column(
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text(title, weight=ft.FontWeight.BOLD, size=14, text_align=ft.TextAlign.CENTER),
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
                        border_radius=20,
                        bgcolor=badge_bg,
                        content=ft.Text(tone, size=12, text_align=ft.TextAlign.CENTER),
                    ),
                    ft.Column(
                        spacing=6,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Container(
                                padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                                border_radius=12,
                                bgcolor=badge_bg,
                                content=ft.Text(f"Signal: {strength}", size=10),
                            ),
                            ft.Container(
                                padding=ft.Padding.symmetric(horizontal=8, vertical=4),
                                border_radius=12,
                                bgcolor=badge_bg,
                                content=ft.Text(f"Confidence: {confidence}", size=10),
                            ),
                        ],
                    ),
                    ft.Text(
                        result.get("feedback", ""),
                        size=10,
                        opacity=0.9,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            ),
        )

    def agreement_card(ml_result: dict, dl_result: dict):
        return card(
            ft.Column(
                spacing=6,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Model insight", weight=ft.FontWeight.BOLD, size=14),
                    ft.Text(
                        model_agreement_status(ml_result, dl_result),
                        text_align=ft.TextAlign.CENTER,
                        size=13,
                    ),
                    ft.Text(
                        compare_model_outputs(ml_result, dl_result),
                        size=12,
                        opacity=0.9,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        confidence_comparison_text(ml_result, dl_result),
                        size=11,
                        opacity=0.8,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            ),
            bordered=False,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
            padding=12,
        )

    # ---------- Home ----------
    def home_screen():
        entries = load_entries_by_date()
        streak = current_entry_streak(entries)
        longest = longest_entry_streak(entries)
        weekly = get_period_stats(entries, 7)
        support_note = support_prompt_text(entries)

        quick_stats = card(
            ft.Column(
                spacing=6,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Quick overview", weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                    ft.Text(f"Current entry streak: {streak} day(s)", text_align=ft.TextAlign.CENTER),
                    ft.Text(f"Longest streak: {longest} day(s)", text_align=ft.TextAlign.CENTER),
                    ft.Text(
                        f"Last 7 days: {weekly['present']} entries | {weekly['pos']} positive | {weekly['neg']} negative",
                        size=12,
                        opacity=0.85,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            )
        )

        footer_text = support_note if support_note else (
            "This journal is designed for supportive reflection and pattern tracking. "
            "It does not provide a diagnosis or medical advice."
        )

        return ft.SafeArea(
            ft.Container(
                expand=True,
                alignment=ft.Alignment(0, 0),
                content=ft.Column(
                    expand=True,
                    width=360,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(
                            expand=True,
                            content=ft.Column(
                                alignment=ft.MainAxisAlignment.CENTER,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                spacing=14,
                                controls=[
                                    ft.Text("Welcome 👋", size=30, weight=ft.FontWeight.BOLD,
                                            text_align=ft.TextAlign.CENTER),
                                    ft.Text(
                                        "Write one entry per day, compare two models, and explore patterns over time.",
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                    quick_stats,
                                    ft.Container(
                                        width=220,
                                        content=ft.FilledButton(
                                            width=220,
                                            content=ft.Text("Journal"),
                                            on_click=lambda _: set_route("/journal"),
                                        ),
                                    ),
                                    ft.Container(
                                        width=220,
                                        content=ft.OutlinedButton(
                                            width=220,
                                            content=ft.Text("Diary"),
                                            on_click=lambda _: set_route("/diary"),
                                        ),
                                    ),
                                    ft.Container(
                                        width=220,
                                        content=ft.OutlinedButton(
                                            width=220,
                                            content=ft.Text("Analysis"),
                                            on_click=lambda _: set_route("/analysis"),
                                        ),
                                    ),
                                    ft.Container(
                                        width=220,
                                        content=ft.TextButton(
                                            content=ft.Text("Settings"),
                                            on_click=lambda _: set_route("/settings"),
                                        ),
                                    ),
                                ],
                            ),
                        ),
                        ft.Container(
                            padding=ft.Padding.only(top=8, bottom=4),
                            content=ft.Text(
                                footer_text,
                                size=11,
                                opacity=0.75,
                                text_align=ft.TextAlign.CENTER,
                            ),
                        ),
                    ],
                ),
            )
        )

    # ---------- Journal ----------
    def journal_screen():
        day = today_iso()
        entries = load_entries_by_date()
        existing = entries.get(day)

        entry_box = ft.TextField(
            label=f"Journal entry — {day}",
            multiline=True,
            min_lines=5,
            max_lines=6,
            width=350,
            value=(existing.get("text") if existing else ""),
            border_radius=16,
            filled=True,
            text_align=ft.TextAlign.LEFT,
        )

        overall_headline = ft.Text(
            "",
            size=17,
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER,
        )

        comparison_text = ft.Text(
            "",
            selectable=True,
            text_align=ft.TextAlign.CENTER,
            size=12,
            opacity=0.92,
        )

        status_text = ft.Text(
            "",
            size=11,
            opacity=0.8,
            text_align=ft.TextAlign.CENTER,
        )

        reflection_text = ft.Text(
            "",
            size=12,
            opacity=0.92,
            text_align=ft.TextAlign.CENTER,
        )

        support_text = ft.Text(
            "",
            size=11,
            opacity=0.85,
            text_align=ft.TextAlign.CENTER,
        )

        # Fixed content holders that always exist
        ml_content = ft.Column(
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Text("Classic model", weight=ft.FontWeight.BOLD, size=14),
                ft.Text(
                    "No result yet.",
                    size=11,
                    opacity=0.7,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
        )

        dl_content = ft.Column(
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Text("Deep model", weight=ft.FontWeight.BOLD, size=14),
                ft.Text(
                    "No result yet.",
                    size=11,
                    opacity=0.7,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
        )

        def set_model_cards(ml_result=None, dl_result=None):
            ml_content.controls = [
                ft.Text("Classic model", weight=ft.FontWeight.BOLD, size=14),
                model_result_block("Classic model", ml_result) if ml_result else ft.Text(
                    "No result yet.",
                    size=11,
                    opacity=0.7,
                    text_align=ft.TextAlign.CENTER,
                ),
            ]

            dl_content.controls = [
                ft.Text("Deep model", weight=ft.FontWeight.BOLD, size=14),
                model_result_block("Deep model", dl_result) if dl_result else ft.Text(
                    "No result yet.",
                    size=11,
                    opacity=0.7,
                    text_align=ft.TextAlign.CENTER,
                ),
            ]

        def populate_from_existing():
            if not existing:
                return

            ml_existing = existing.get("sentiment_ml")
            dl_existing = existing.get("sentiment_dl")

            if ml_existing is None and isinstance(existing.get("sentiment"), dict):
                ml_existing = existing.get("sentiment")

            if ml_existing:
                overall_headline.value = f"Overall tone: {tone_from_result(ml_existing)}"

            if ml_existing and dl_existing:
                comparison_text.value = compare_model_outputs(ml_existing, dl_existing)
                reflection_text.value = reflection_prompt_for_entry(ml_existing, dl_existing)
                support_text.value = support_prompt_text(entries)

            set_model_cards(ml_existing, dl_existing)

        populate_from_existing()

        def analyze_and_save(_):
            text = (entry_box.value or "").strip()

            if len(text) < 5:
                status_text.value = "Please write a little more before analyzing."
                overall_headline.value = ""
                comparison_text.value = ""
                reflection_text.value = ""
                support_text.value = ""
                set_model_cards(None, None)
                page.update()
                return

            try:
                ml_result = predict_with_classic_model(text)
            except Exception as ex:
                ml_result = {
                    "label": "unknown",
                    "predicted_class": None,
                    "p_negative": None,
                    "threshold": None,
                    "feedback": f"Classic model error: {ex}",
                    "model": "classic-ml",
                    "available": False,
                }

            dl_result = predict_with_deep_model(text)

            overall_headline.value = f"Overall tone: {tone_from_result(ml_result)}"
            comparison_text.value = compare_model_outputs(ml_result, dl_result)
            reflection_text.value = reflection_prompt_for_entry(ml_result, dl_result)

            entry_obj = {
                "day": day,
                "timestamp": now_iso(),
                "text": text,
                "sentiment_ml": ml_result,
                "sentiment_dl": dl_result,
            }
            upsert_entry_by_date(day, entry_obj)

            updated_entries = load_entries_by_date()
            support_text.value = support_prompt_text(updated_entries)

            set_model_cards(ml_result, dl_result)

            status_text.value = "Analyzed and saved."
            page.update()

        return ft.SafeArea(
            ft.Container(
                expand=True,
                alignment=ft.Alignment(0, -1),
                content=ft.Column(
                    width=390,
                    spacing=12,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        top_bar("Journal"),

                        card(
                            ft.Column(
                                spacing=10,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    ft.Text(
                                        "Write about today",
                                        size=18,
                                        weight=ft.FontWeight.BOLD,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                    ft.Text(
                                        "Capture your day, then compare how both models interpret the tone.",
                                        size=12,
                                        opacity=0.78,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                    entry_box,
                                    ft.Row(
                                        alignment=ft.MainAxisAlignment.CENTER,
                                        spacing=10,
                                        controls=[
                                            ft.FilledButton(
                                                content=ft.Text("Analyze & Save"),
                                                width=210,
                                                height=44,
                                                on_click=analyze_and_save,
                                            ),
                                        ],
                                    ),
                                    status_text,
                                ],
                            ),
                            bordered=False,
                            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.WHITE),
                            padding=16,
                        ),

                        card(
                            ft.Column(
                                spacing=8,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    ft.Text(
                                        "Today’s result",
                                        size=15,
                                        weight=ft.FontWeight.BOLD,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                    overall_headline if overall_headline.value else ft.Text(
                                        "No result yet",
                                        size=16,
                                        weight=ft.FontWeight.BOLD,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                    comparison_text if comparison_text.value else ft.Text(
                                        "Your comparison will appear here after analysis.",
                                        size=12,
                                        opacity=0.7,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                ],
                            ),
                            bordered=False,
                            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.WHITE),
                            padding=14,
                        ),

                        ft.Row(
                            width=390,
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=10,
                            vertical_alignment=ft.CrossAxisAlignment.START,
                            controls=[
                                ft.Container(
                                    width=190,
                                    content=card(
                                        ml_content,
                                        bordered=True,
                                        padding=12,
                                    ),
                                ),
                                ft.Container(
                                    width=190,
                                    content=card(
                                        dl_content,
                                        bordered=True,
                                        padding=12,
                                    ),
                                ),
                            ],
                        ),

                        card(
                            ft.Column(
                                spacing=6,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    ft.Text("Reflection prompt", weight=ft.FontWeight.BOLD, size=14),
                                    ft.Text(
                                        reflection_text.value if reflection_text.value else
                                        "After analysis, a short reflection prompt will appear here.",
                                        size=12,
                                        opacity=0.9,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                ],
                            ),
                            bordered=False,
                            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.WHITE),
                            padding=12,
                        ),

                        ft.Container(
                            width=370,
                            padding=ft.Padding.only(top=2),
                            content=ft.Text(
                                support_text.value if support_text.value else
                                "This app highlights language patterns in journal writing. It is a reflection tool only and does not diagnose depression or any mental health condition.",
                                size=11,
                                opacity=0.72,
                                text_align=ft.TextAlign.CENTER,
                            ),
                        ),
                    ],
                ),
            )
        )

    # ---------- Diary ----------
    def diary_screen():
        entries = load_entries_by_date()
        available_days = sorted(entries.keys())

        selected_date = ft.Text("No date selected.", size=13, opacity=0.8)
        entry_text = ft.Text("Tap a day on the calendar to view an entry.", selectable=True)

        ml_display = ft.Column()
        dl_display = ft.Column()
        comparison_display = ft.Column()
        editable_box = ft.TextField(
            label="Edit entry",
            multiline=True,
            min_lines=6,
            max_lines=8,
            visible=False,
            filled=True,
            border_radius=12,
        )
        edit_status = ft.Text("", size=12, opacity=0.8)
        edit_buttons = ft.Row(
            visible=False,
            alignment=ft.MainAxisAlignment.CENTER,
            controls=[],
        )

        view_year = date.today().year
        view_month = date.today().month
        selected_day_iso = None

        month_title = ft.Text("", size=16, weight=ft.FontWeight.BOLD)
        calendar_grid = ft.Column(spacing=6)

        def diary_day_model(day_iso: str):
            obj = entries.get(day_iso)
            if not obj:
                return None

            ml = obj.get("sentiment_ml")
            if ml:
                return ml

            legacy = obj.get("sentiment")
            if legacy:
                return legacy

            return None

        def day_bg(day_iso: str):
            s = diary_day_model(day_iso)
            if not s:
                return ft.Colors.GREY_200
            cls = label_to_sentiment_class(s.get("label", ""))
            if cls == "neg":
                return ft.Colors.RED_300
            if cls == "pos":
                return ft.Colors.GREEN_300
            return ft.Colors.AMBER_300

        def rebuild_month_calendar():
            nonlocal entries, available_days
            entries = load_entries_by_date()
            available_days = sorted(entries.keys())

            first = date(view_year, view_month, 1)
            month_title.value = first.strftime("%B %Y")
            calendar_grid.controls.clear()

            header = ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Container(width=48, alignment=ft.Alignment(0, 0), content=ft.Text("Mon", size=12, weight=ft.FontWeight.BOLD)),
                    ft.Container(width=48, alignment=ft.Alignment(0, 0), content=ft.Text("Tue", size=12, weight=ft.FontWeight.BOLD)),
                    ft.Container(width=48, alignment=ft.Alignment(0, 0), content=ft.Text("Wed", size=12, weight=ft.FontWeight.BOLD)),
                    ft.Container(width=48, alignment=ft.Alignment(0, 0), content=ft.Text("Thu", size=12, weight=ft.FontWeight.BOLD)),
                    ft.Container(width=48, alignment=ft.Alignment(0, 0), content=ft.Text("Fri", size=12, weight=ft.FontWeight.BOLD)),
                    ft.Container(width=48, alignment=ft.Alignment(0, 0), content=ft.Text("Sat", size=12, weight=ft.FontWeight.BOLD)),
                    ft.Container(width=48, alignment=ft.Alignment(0, 0), content=ft.Text("Sun", size=12, weight=ft.FontWeight.BOLD)),
                ],
            )
            calendar_grid.controls.append(header)

            cal = calendar.Calendar(firstweekday=0)
            weeks = cal.monthdayscalendar(view_year, view_month)
            today = date.today().isoformat()

            for week in weeks:
                row = []
                for d in week:
                    if d == 0:
                        row.append(ft.Container(width=48, height=42))
                        continue

                    day_iso = date(view_year, view_month, d).isoformat()
                    bg = day_bg(day_iso)

                    border = None
                    if day_iso == selected_day_iso:
                        border = ft.Border.all(2, ft.Colors.BLUE_400)
                    elif day_iso == today:
                        border = ft.Border.all(1, ft.Colors.BLUE_200)

                    cell = ft.Container(
                        width=48,
                        height=42,
                        border_radius=10,
                        bgcolor=bg,
                        border=border,
                        alignment=ft.Alignment(0, 0),
                        content=ft.Text(str(d), size=12, weight=ft.FontWeight.BOLD),
                    )

                    row.append(
                        ft.GestureDetector(
                            on_tap=lambda e, day=day_iso: show_day(day),
                            content=cell,
                        )
                    )

                calendar_grid.controls.append(
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=row,
                    )
                )

        def save_edited_day(_):
            nonlocal selected_day_iso
            if not selected_day_iso:
                return
            text = (editable_box.value or "").strip()
            if len(text) < 5:
                edit_status.value = "Please write a little more before saving."
                page.update()
                return

            try:
                ml_result = predict_with_classic_model(text)
            except Exception as ex:
                ml_result = {
                    "label": "unknown",
                    "predicted_class": None,
                    "p_negative": None,
                    "threshold": None,
                    "feedback": f"Classic model error: {ex}",
                    "model": "classic-ml",
                    "available": False,
                }

            dl_result = predict_with_deep_model(text)

            obj = {
                "day": selected_day_iso,
                "timestamp": now_iso(),
                "text": text,
                "sentiment_ml": ml_result,
                "sentiment_dl": dl_result,
            }
            upsert_entry_by_date(selected_day_iso, obj)
            edit_status.value = "Entry updated."
            show_day(selected_day_iso)

        def delete_selected_day(_):
            nonlocal selected_day_iso
            if not selected_day_iso:
                return
            delete_entry_by_date(selected_day_iso)
            edit_status.value = "Entry deleted."
            selected_date.value = f"Viewing: {selected_day_iso}"
            entry_text.value = "No journal entry saved for this date."
            ml_display.controls = []
            dl_display.controls = []
            comparison_display.controls = []
            editable_box.visible = False
            edit_buttons.visible = False
            rebuild_month_calendar()
            page.update()

        def show_day(day: str):
            nonlocal entries, available_days, selected_day_iso
            entries = load_entries_by_date()
            available_days = sorted(entries.keys())

            selected_day_iso = day
            selected_date.value = f"Viewing: {friendly_date_text(day)}"
            edit_status.value = ""

            obj = entries.get(day)
            if not obj:
                entry_text.value = "No journal entry saved for this date."
                ml_display.controls = []
                dl_display.controls = []
                comparison_display.controls = []
                editable_box.visible = False
                edit_buttons.visible = False
            else:
                entry_text.value = obj.get("text", "")
                editable_box.value = obj.get("text", "")
                editable_box.visible = True

                ml_result = obj.get("sentiment_ml")
                dl_result = obj.get("sentiment_dl")

                if ml_result is None and isinstance(obj.get("sentiment"), dict):
                    ml_result = obj.get("sentiment")

                ml_display.controls = [model_result_block("Classic model", ml_result)] if ml_result else []
                dl_display.controls = [model_result_block("Deep model", dl_result)] if dl_result else []

                comparison_controls = []
                if ml_result and dl_result:
                    comparison_controls.append(agreement_card(ml_result, dl_result))
                    comparison_controls.append(
                        card(
                            ft.Column(
                                spacing=6,
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    ft.Text("Reflection prompt", weight=ft.FontWeight.BOLD),
                                    ft.Text(
                                        reflection_prompt_for_entry(ml_result, dl_result),
                                        size=12,
                                        opacity=0.9,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                ],
                            )
                        )
                    )
                comparison_display.controls = comparison_controls

                edit_buttons.controls = [
                    ft.FilledButton(content=ft.Text("Save changes"), on_click=save_edited_day),
                    ft.OutlinedButton(content=ft.Text("Delete entry"), on_click=delete_selected_day),
                ]
                edit_buttons.visible = True

            rebuild_month_calendar()
            page.update()

        def normalize_day(v) -> str:
            if v is None:
                return ""
            if hasattr(v, "date"):
                return v.date().isoformat()
            if hasattr(v, "isoformat"):
                return v.isoformat()
            return str(v)[:10]

        date_picker = ft.DatePicker()

        def on_date_change(_):
            if not date_picker.value:
                return
            day = normalize_day(date_picker.value)
            try:
                dt = datetime.strptime(day, "%Y-%m-%d").date()
                nonlocal view_year, view_month
                view_year, view_month = dt.year, dt.month
            except Exception:
                pass
            show_day(day)

        date_picker.on_change = on_date_change
        page.overlay.clear()
        page.overlay.append(date_picker)

        def open_calendar_picker(_):
            date_picker.open = True
            page.update()

        def prev_month(_):
            nonlocal view_year, view_month
            view_month -= 1
            if view_month == 0:
                view_month = 12
                view_year -= 1
            rebuild_month_calendar()
            page.update()

        def next_month(_):
            nonlocal view_year, view_month
            view_month += 1
            if view_month == 13:
                view_month = 1
                view_year += 1
            rebuild_month_calendar()
            page.update()

        rebuild_month_calendar()

        calendar_view = ft.Column(
            spacing=8,
            controls=[
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.TextButton(content=ft.Text("Prev"), on_click=prev_month),
                        month_title,
                        ft.TextButton(content=ft.Text("Next"), on_click=next_month),
                    ],
                ),
                ft.Container(
                    padding=10,
                    border_radius=12,
                    border=ft.Border.all(1, ft.Colors.GREY_300),
                    content=calendar_grid,
                ),
                ft.Row(
                    spacing=12,
                    controls=[
                        ft.Row(spacing=6, controls=[
                            ft.Container(width=12, height=12, bgcolor=ft.Colors.GREEN_300, border_radius=3),
                            ft.Text("Positive tone", size=12, opacity=0.8),
                        ]),
                        ft.Row(spacing=6, controls=[
                            ft.Container(width=12, height=12, bgcolor=ft.Colors.RED_300, border_radius=3),
                            ft.Text("Negative tone", size=12, opacity=0.8),
                        ]),
                    ],
                ),
            ],
        )

        return ft.SafeArea(
            ft.Column(
                expand=True,
                spacing=12,
                controls=[
                    top_bar("Diary"),
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.FilledButton(content=ft.Text("Pick date"), on_click=open_calendar_picker),
                            ft.Text(f"{len(available_days)} saved", opacity=0.8),
                        ],
                    ),
                    calendar_view,
                    selected_date,
                    ft.Text("Saved entry", size=16, weight=ft.FontWeight.BOLD),
                    card(entry_text),
                    ft.Text("Edit selected entry", size=16, weight=ft.FontWeight.BOLD),
                    card(
                        ft.Column(
                            spacing=8,
                            controls=[
                                editable_box,
                                edit_buttons,
                                edit_status,
                            ],
                        )
                    ),
                    ft.Row(
                        spacing=12,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                        controls=[
                            ft.Container(expand=True, content=ml_display),
                            ft.Container(expand=True, content=dl_display),
                        ],
                    ),
                    comparison_display,
                ],
            )
        )

    # ---------- Analysis ----------
    def analysis_screen():
        entries = load_entries_by_date()

        end_day = date.today()
        start_day = end_day - timedelta(days=29)
        days = [start_day + timedelta(days=i) for i in range(30)]
        keys = [d.isoformat() for d in days]

        ml_points = []
        dl_points = []
        ml_pvals = []
        dl_pvals = []
        agreements = 0
        disagreements = 0

        for i, k in enumerate(keys):
            obj = entries.get(k)
            if not obj:
                continue

            ml = obj.get("sentiment_ml")
            if ml is None and isinstance(obj.get("sentiment"), dict):
                ml = obj.get("sentiment")

            dl = obj.get("sentiment_dl")

            if ml is not None:
                p = get_p_negative(ml)
                if p is not None:
                    try:
                        p = float(p)
                        ml_pvals.append(p)
                        ml_points.append((i, p, k))
                    except Exception:
                        pass

            if dl is not None and dl.get("available", True):
                p = get_p_negative(dl)
                if p is not None:
                    try:
                        p = float(p)
                        dl_pvals.append(p)
                        dl_points.append((i, p, k))
                    except Exception:
                        pass

            if ml and dl and dl.get("available", False):
                if normalize_label(ml.get("label")) == normalize_label(dl.get("label")):
                    agreements += 1
                else:
                    disagreements += 1

        weekly = get_period_stats(entries, 7)
        monthly = get_period_stats(entries, 30)
        streak = current_entry_streak(entries)
        longest = longest_entry_streak(entries)
        support_note = support_prompt_text(entries)

        def build_combined_mood_arc():
            if not ml_points and not dl_points:
                return ft.Text("No analysis data yet for the last 30 days.", opacity=0.85)

            chart_height = 260
            label_col_width = 70
            point_width = 12
            point_gap = 4
            line_thickness = 4

            ml_map = {x: y for x, y, _ in ml_points}
            dl_map = {x: y for x, y, _ in dl_points}

            def y_to_px(v: float) -> int:
                return int((1 - v) * (chart_height - 20)) + 10

            plot_width = 30 * (point_width + point_gap)
            plot_stack = ft.Stack(width=plot_width, height=chart_height)

            for frac in [0.0, 0.5, 1.0]:
                y = y_to_px(frac)
                plot_stack.controls.append(
                    ft.Container(
                        left=0,
                        top=y,
                        width=plot_width,
                        height=1,
                        bgcolor=ft.Colors.GREY_300,
                    )
                )

            def add_series(points, p_map, line_color, point_color, tooltip_prefix, point_size):
                present_indices = sorted([x for x, _, _ in points])

                for a, b in zip(present_indices[:-1], present_indices[1:]):
                    if b != a + 1:
                        continue

                    y1 = y_to_px(p_map[a])
                    y2 = y_to_px(p_map[b])
                    x1 = a * (point_width + point_gap) + point_width // 2
                    x2 = b * (point_width + point_gap) + point_width // 2

                    dx = x2 - x1
                    dy = y2 - y1
                    length = max(1, int((dx * dx + dy * dy) ** 0.5))
                    angle = np.degrees(np.arctan2(dy, dx))

                    plot_stack.controls.append(
                        ft.Container(
                            left=x1,
                            top=y1 - line_thickness // 2,
                            width=length,
                            height=line_thickness,
                            bgcolor=line_color,
                            border_radius=4,
                            rotate=ft.Rotate(
                                angle=angle * 3.1415926535 / 180,
                                alignment=ft.Alignment(-1, 0),
                            ),
                        )
                    )

                for x, y, k in points:
                    px = x * (point_width + point_gap)
                    py = y_to_px(y) - (point_size // 2)
                    pretty_date = friendly_date_text(k)
                    tooltip = (
                        f"{tooltip_prefix}\n"
                        f"Date: {pretty_date}\n"
                        f"Tone: {tone_from_probability(y)}\n"
                        f"p_negative: {y:.2f}"
                    )
                    plot_stack.controls.append(
                        ft.Container(
                            left=px,
                            top=py,
                            width=point_size,
                            height=point_size,
                            bgcolor=point_color,
                            border_radius=point_size,
                            tooltip=tooltip,
                        )
                    )

            add_series(
                points=ml_points,
                p_map=ml_map,
                line_color=ft.Colors.BLUE_400,
                point_color=ft.Colors.BLUE_500,
                tooltip_prefix="Classic model",
                point_size=12,
            )

            add_series(
                points=dl_points,
                p_map=dl_map,
                line_color=ft.Colors.ORANGE_400,
                point_color=ft.Colors.ORANGE_500,
                tooltip_prefix="Deep model",
                point_size=10,
            )

            y_axis = ft.Container(
                width=label_col_width,
                height=chart_height,
                padding=ft.Padding.only(right=10),
                alignment=ft.Alignment(1, 0),
                content=ft.Column(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    horizontal_alignment=ft.CrossAxisAlignment.END,
                    controls=[
                        ft.Text("Negative\ntone", size=11, text_align=ft.TextAlign.RIGHT),
                        ft.Text("Mixed", size=11, text_align=ft.TextAlign.RIGHT),
                        ft.Text("Positive\ntone", size=11, text_align=ft.TextAlign.RIGHT),
                    ],
                ),
            )

            graph_row = ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.START,
                controls=[
                    y_axis,
                    ft.Container(
                        padding=ft.Padding.only(left=6, right=6, top=10),
                        content=plot_stack,
                    ),
                ],
            )

            x_labels = ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[
                    ft.Container(width=label_col_width),
                    ft.Container(
                        width=plot_width + 12,
                        padding=ft.Padding.only(top=14),
                        content=ft.Row(
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            controls=[
                                ft.Text(days[0].strftime("%d %b"), size=10, opacity=0.8),
                                ft.Text(days[7].strftime("%d %b"), size=10, opacity=0.8),
                                ft.Text(days[14].strftime("%d %b"), size=10, opacity=0.8),
                                ft.Text(days[21].strftime("%d %b"), size=10, opacity=0.8),
                                ft.Text(days[29].strftime("%d %b"), size=10, opacity=0.8),
                            ],
                        ),
                    ),
                ],
            )

            legend = ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20,
                controls=[
                    ft.Row(
                        spacing=6,
                        controls=[
                            ft.Container(width=18, height=4, bgcolor=ft.Colors.BLUE_400, border_radius=2),
                            ft.Text("Classic model", size=12, opacity=0.85),
                        ],
                    ),
                    ft.Row(
                        spacing=6,
                        controls=[
                            ft.Container(width=18, height=4, bgcolor=ft.Colors.ORANGE_400, border_radius=2),
                            ft.Text("Deep model", size=12, opacity=0.85),
                        ],
                    ),
                ],
            )

            return ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
                controls=[
                    legend,
                    graph_row,
                    x_labels,
                ],
            )

        chart = build_combined_mood_arc()

        weekly_card = card(
            ft.Column(
                spacing=6,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Weekly summary", weight=ft.FontWeight.BOLD),
                    ft.Text(
                        f"{weekly['present']} entries | {weekly['pos']} positive | {weekly['neg']} negative | {weekly['missing']} missing",
                        size=12,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        f"Average tone: {average_tone_text(weekly['avg_p'])}",
                        size=12,
                        opacity=0.85,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            )
        )

        monthly_card = card(
            ft.Column(
                spacing=6,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Monthly summary", weight=ft.FontWeight.BOLD),
                    ft.Text(
                        f"{monthly['present']} entries | {monthly['pos']} positive | {monthly['neg']} negative | {monthly['missing']} missing",
                        size=12,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        f"Average tone: {average_tone_text(monthly['avg_p'])}",
                        size=12,
                        opacity=0.85,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            )
        )

        insight_card = card(
            ft.Column(
                spacing=6,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Insights", weight=ft.FontWeight.BOLD),
                    ft.Text(f"Current streak: {streak} day(s)", text_align=ft.TextAlign.CENTER),
                    ft.Text(f"Longest streak: {longest} day(s)", text_align=ft.TextAlign.CENTER),
                    ft.Text(
                        f"Model agreement days: {agreements} | disagreement days: {disagreements}",
                        size=12,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        "Agreement suggests the two models are interpreting entries similarly. Disagreement can point to more mixed language or uncertainty.",
                        size=12,
                        opacity=0.85,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            )
        )

        note = ft.Text(
            "Higher points suggest a more negative tone, while lower points suggest a more positive tone.\n"
            "Blue represents the classic machine learning model and orange represents the deep learning model.\n"
            "If the two lines stay close together, both models are interpreting your entries similarly. Bigger gaps suggest disagreement.",
            size=13,
            opacity=0.82,
            text_align=ft.TextAlign.CENTER,
        )

        disclaimer = card(
            ft.Text(
                support_note if support_note else
                "Important note: this app highlights emotional tone patterns in writing. It is not a medical assessment tool and should not be used as a diagnosis.",
                size=12,
                opacity=0.9,
                text_align=ft.TextAlign.CENTER,
            )
        )

        return ft.SafeArea(
            ft.Column(
                expand=True,
                spacing=16,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    top_bar("Analysis"),
                    ft.Container(
                        alignment=ft.Alignment(0, 0),
                        content=card(
                            ft.Column(
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                spacing=14,
                                controls=[
                                    ft.Text("Mood Arc (last 30 days)", weight=ft.FontWeight.BOLD, size=18),
                                    ft.Text(
                                        "A visual comparison of how both models interpret your journal tone over time.",
                                        size=12,
                                        opacity=0.8,
                                        text_align=ft.TextAlign.CENTER,
                                    ),
                                    chart,
                                ],
                            )
                        ),
                    ),
                    ft.Row(
                        spacing=10,
                        controls=[
                            ft.Container(expand=True, content=weekly_card),
                            ft.Container(expand=True, content=monthly_card),
                        ],
                    ),
                    insight_card,
                    card(
                        ft.Column(
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            spacing=8,
                            controls=[
                                ft.Text("How to read this", weight=ft.FontWeight.BOLD),
                                note,
                            ],
                        )
                    ),
                    disclaimer,
                ],
            )
        )

    # ---------- Settings ----------
    def settings_screen():
        theme_switch = ft.Switch(label="Dark mode", value=True)

        def toggle_theme(_):
            page.theme_mode = ft.ThemeMode.DARK if theme_switch.value else ft.ThemeMode.LIGHT
            page.update()

        theme_switch.on_change = toggle_theme

        return ft.SafeArea(
            ft.Column(
                expand=True,
                spacing=12,
                controls=[
                    top_bar("Settings"),
                    card(theme_switch),
                    card(
                        ft.Text(
                            "Ethical note: this app is designed for reflection, journaling, and pattern tracking. It does not diagnose depression or any mental health condition.",
                            size=13,
                            opacity=0.85,
                        )
                    ),
                    card(
                        ft.Text(
                            "If entries repeatedly suggest a difficult tone, the app may show a gentle support note. That message is only a wellbeing reminder, not medical advice.",
                            size=13,
                            opacity=0.85,
                        )
                    ),
                    card(
                        ft.Text(
                            "To enable deep model comparison, place a local Hugging Face text-classification model in models/distilbert_depression_classifier/",
                            size=13,
                            opacity=0.85,
                        )
                    ),
                ],
            )
        )

    def not_found_screen():
        return ft.SafeArea(
            ft.Column(
                expand=True,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Text("Page not found"),
                    ft.FilledButton(content=ft.Text("Go home"), on_click=lambda _: set_route("/")),
                ],
            )
        )

    if not page.route:
        page.route = "/"
    render()


ft.run(main)