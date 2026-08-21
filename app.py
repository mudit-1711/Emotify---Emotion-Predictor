import streamlit as st
import pickle
import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotify — Emotion Predictor",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── NLTK ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)

download_nltk()

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("lmodel.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("emotions_map.pkl", "rb") as f:
        emotions_map = pickle.load(f)
    return model, tfidf, emotions_map

model, tfidf, emotions_map = load_models()
idx_to_emotion = {v: k for k, v in emotions_map.items()}

# ─── Emotion Config ───────────────────────────────────────────────────────────
EMOTION_META = {
    "joy":      {"emoji": "😄", "color": "#F59E0B", "card_bg": "linear-gradient(145deg,#3d2e00,#5a4500)", "bar": "#F59E0B", "label": "JOY"},
    "sadness":  {"emoji": "😢", "color": "#4F8EF9", "card_bg": "linear-gradient(145deg,#0d1f3c,#1a3a6e)", "bar": "#F97316", "label": "SADNESS"},
    "anger":    {"emoji": "😡", "color": "#EF4444", "card_bg": "linear-gradient(145deg,#3c0d0d,#6e1a1a)", "bar": "#EF4444", "label": "ANGER"},
    "fear":     {"emoji": "😨", "color": "#A855F7", "card_bg": "linear-gradient(145deg,#1e0d3c,#361a6e)", "bar": "#A855F7", "label": "FEAR"},
    "love":     {"emoji": "❤️",  "color": "#EC4899", "card_bg": "linear-gradient(145deg,#3c0d20,#6e1a3c)", "bar": "#EC4899", "label": "LOVE"},
    "surprise": {"emoji": "😲", "color": "#10B981", "card_bg": "linear-gradient(145deg,#0d3c22,#1a6e3c)", "bar": "#10B981", "label": "SURPRISE"},
}

QUOTES = {
    "sadness":  '"Heavy hearts, like heavy clouds in the sky, are best relieved by the letting of a little water." — Antoine de Saint-Exupéry',
    "joy":      '"The most important thing is to enjoy your life — to be happy — it\'s all that matters." — Audrey Hepburn',
    "anger":    '"Speak when you are angry and you will make the best speech you will ever regret." — Ambrose Bierce',
    "fear":     '"Do one thing every day that scares you." — Eleanor Roosevelt',
    "love":     '"The best thing to hold onto in life is each other." — Audrey Hepburn',
    "surprise": '"The appearance of things changes according to the emotions." — Kahlil Gibran',
}

# ─── NLP Pipeline ─────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

def predict(text):
    cleaned = preprocess(text)
    vec = tfidf.transform([cleaned])
    pred_idx = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    return idx_to_emotion[pred_idx], proba, cleaned

def get_keywords(cleaned, top_n=5):
    feature_names = tfidf.get_feature_names_out()
    vec = tfidf.transform([cleaned])
    pred_idx = model.predict(vec)[0]
    scores = model.coef_[pred_idx] * vec.toarray()[0]
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [feature_names[i] for i in top_idx if scores[i] > 0]

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] { background: #0d0d0d !important; }
.main .block-container { padding: 28px 32px 16px 32px !important; max-width: 100% !important; }
header[data-testid="stHeader"], [data-testid="stToolbar"], footer, #MainMenu { display: none !important; }

.emo-title {
    font-family: 'Poppins', sans-serif !important;
    font-size: 52px; font-weight: 900;
    background: linear-gradient(135deg, #ff2d78 0%, #ff6b35 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1; margin-bottom: 6px;
}
.emo-tagline { font-size: 13px; color: rgba(255,255,255,0.45); margin-bottom: 6px; }
.emo-model-badge { font-size: 12px; color: rgba(255,255,255,0.3); }
.input-label { font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.55); margin-bottom: 8px; margin-top: 16px; }

textarea {
    background: #1a1a1a !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.8) !important;
    font-size: 14px !important;
    resize: none !important; padding: 12px !important;
}
textarea:focus {
    border-color: rgba(255,45,120,0.35) !important;
    box-shadow: 0 0 0 3px rgba(255,45,120,0.08) !important;
    outline: none !important;
}
textarea::placeholder { color: rgba(255,255,255,0.2) !important; }
[data-testid="stTextArea"] label { display: none !important; }

[data-testid="stButton"] > button {
    width: 100% !important; padding: 12px 20px !important;
    border-radius: 10px !important; background: #1a1a1a !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.75) !important;
    font-size: 14px !important; font-weight: 600 !important;
    cursor: pointer !important; transition: all 0.2s !important;
    box-shadow: none !important;
}
[data-testid="stButton"] > button:hover {
    background: #222 !important;
    border-color: rgba(255,45,120,0.4) !important;
    color: #fff !important;
    box-shadow: 0 0 12px rgba(255,45,120,0.15) !important;
}
[data-testid="stButton"] > button:focus { box-shadow: none !important; outline: none !important; }

.empty-card {
    background: #141414; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 50px 30px; text-align: center;
    color: rgba(255,255,255,0.2); min-height: 260px;
}
.e-icon { font-size: 44px; margin-bottom: 12px; filter: grayscale(1) opacity(0.35); }
.e-title { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
.e-sub { font-size: 12px; line-height: 1.7; max-width: 220px; }

[data-testid="column"] { padding: 0 8px !important; }
.stMarkdown { margin-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="emo-title">Emotify</div>
<div class="emo-tagline">Discover the underlying emotions in your writing using machine learning</div>
<div class="emo-model-badge">👤 Active Model: Logistic Regression (Accuracy: 86.2%)</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

# ─── THREE COLUMN LAYOUT ──────────────────────────────────────────────────────
col_input, col_result, col_probs = st.columns([1.1, 1.3, 1.1], gap="medium")

# ══════════════════════════ LEFT: INPUT ═══════════════════════════════════════
with col_input:
    st.markdown('<div class="input-label">Express your thoughts</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        label="text",
        placeholder="Type what's on your mind…",
        height=200,
        key="user_text",
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("✨  Analyze Emotion", use_container_width=True)

    if analyze_clicked:
        if user_input.strip():
            emotion, proba, cleaned = predict(user_input)
            keywords = get_keywords(cleaned)
            st.session_state.result = {
                "emotion": emotion,
                "proba": proba,
                "keywords": keywords,
            }
        else:
            st.session_state.result = "empty"

# ══════════════════════════ MIDDLE: RESULT ════════════════════════════════════
with col_result:
    res = st.session_state.result

    if res is None or res == "empty":
        msg = "Enter text and click Analyze Emotion to see results." if res == "empty" else "Your emotion analysis will appear here."
        st.markdown(
            '<div class="empty-card">'
            '<div class="e-icon">🎭</div>'
            '<div class="e-title">Analysis Result</div>'
            '<div class="e-sub">' + msg + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        emotion  = res["emotion"]
        proba    = res["proba"]
        keywords = res["keywords"]
        meta     = EMOTION_META.get(emotion, EMOTION_META["joy"])
        quote    = QUOTES.get(emotion, "")
        card_bg  = meta["card_bg"]
        emoji    = meta["emoji"]
        label    = meta["label"]

        # Build keyword chips as plain string (no f-string nesting)
        if keywords:
            kw_chips = ""
            for kw in keywords:
                kw_chips += (
                    '<span style="display:inline-flex;align-items:center;gap:4px;'
                    'padding:4px 10px;border-radius:20px;'
                    'background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.2);'
                    'color:rgba(245,158,11,0.9);font-size:12px;font-weight:500;margin:2px;">🔑 '
                    + kw + '</span>'
                )
        else:
            kw_chips = '<span style="color:rgba(255,255,255,0.2);font-size:12px;">No key signals detected</span>'

        # Render card header + emotion card
        st.markdown(
            '<div style="background:#141414;border:1px solid rgba(255,255,255,0.08);border-radius:14px;overflow:hidden;">'
            '<div style="font-size:12px;font-weight:600;color:rgba(255,255,255,0.4);letter-spacing:0.8px;'
            'padding:14px 18px 10px 18px;border-bottom:1px solid rgba(255,255,255,0.06);">Analysis Result</div>',
            unsafe_allow_html=True,
        )
        # Emotion card with gradient background
        st.markdown(
            '<div style="background:' + card_bg + ';border-radius:12px;padding:28px 20px;'
            'display:flex;flex-direction:column;align-items:center;justify-content:center;'
            'margin:14px;text-align:center;min-height:160px;">'
            '<div style="font-size:52px;margin-bottom:12px;line-height:1;">' + emoji + '</div>'
            '<div style="font-family:Poppins,sans-serif;font-size:26px;font-weight:900;'
            'color:#ffffff;letter-spacing:2px;">' + label + '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Quote
        st.markdown(
            '<div style="font-size:12px;font-style:italic;color:rgba(255,255,255,0.4);'
            'text-align:center;line-height:1.6;padding:0 14px 14px 14px;">' + quote + '</div>',
            unsafe_allow_html=True,
        )
        # Keywords section
        st.markdown(
            '<div style="padding:10px 14px 14px 14px;border-top:1px solid rgba(255,255,255,0.06);">'
            '<div style="font-size:11px;font-weight:600;color:rgba(255,255,255,0.35);margin-bottom:8px;">Key Words:</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:7px;">' + kw_chips + '</div>'
            '</div>'
            '</div>',   # closes the outer card wrapper div
            unsafe_allow_html=True,
        )

# ══════════════════════════ RIGHT: PROBABILITIES ══════════════════════════════
with col_probs:
    res = st.session_state.result

    if res is None or res == "empty":
        st.markdown(
            '<div class="empty-card">'
            '<div class="e-icon">📊</div>'
            '<div class="e-title">Emotion Probabilities</div>'
            '<div class="e-sub">Probability breakdown will appear here after analysis.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        proba = res["proba"]

        # Card header (opened div — no f-string)
        st.markdown(
            '<div style="background:#141414;border:1px solid rgba(255,255,255,0.08);'
            'border-radius:14px;overflow:hidden;">'
            '<div style="font-size:12px;font-weight:600;color:rgba(255,255,255,0.4);'
            'letter-spacing:0.8px;padding:14px 18px 10px 18px;'
            'border-bottom:1px solid rgba(255,255,255,0.06);">Emotion Probabilities</div>',
            unsafe_allow_html=True,
        )

        # Each row is a SEPARATE st.markdown call — no HTML string embedding
        sorted_emotions = sorted(emotions_map.items(), key=lambda x: -proba[x[1]])
        last_idx = len(sorted_emotions) - 1
        for i, (emo_name, emo_idx) in enumerate(sorted_emotions):
            m     = EMOTION_META.get(emo_name, {"emoji": "🔵", "bar": "#888", "label": emo_name.title()})
            pct   = proba[emo_idx] * 100
            emoji = m["emoji"]
            lbl   = m["label"].capitalize()
            bar   = m["bar"]
            bd    = "none" if i == last_idx else "1px solid rgba(255,255,255,0.04)"
            w     = "{:.1f}".format(pct)

            st.markdown(
                '<div style="display:flex;align-items:center;gap:10px;padding:9px 18px;border-bottom:' + bd + ';">'
                '<span style="font-size:15px;width:22px;text-align:center;">' + emoji + '</span>'
                '<span style="font-size:13px;color:rgba(255,255,255,0.65);width:70px;">' + lbl + '</span>'
                '<div style="flex:1;height:6px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">'
                '<div style="width:' + w + '%;height:100%;background:' + bar + ';border-radius:99px;"></div>'
                '</div>'
                '<span style="font-size:12px;font-weight:600;color:rgba(255,255,255,0.5);width:38px;text-align:right;">'
                + w + '%</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Close card wrapper div
        st.markdown("</div>", unsafe_allow_html=True)