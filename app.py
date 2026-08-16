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

# ─── Download NLTK data silently ──────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

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

# Reverse map: int → emotion name
idx_to_emotion = {v: k for k, v in emotions_map.items()}

# ─── Emotion Metadata ─────────────────────────────────────────────────────────
EMOTION_META = {
    "joy":      {"emoji": "😄", "color": "#F9C74F", "bg": "#fffbe6", "label": "Joy"},
    "sadness":  {"emoji": "😢", "color": "#4F8EF9", "bg": "#e8f0fe", "label": "Sadness"},
    "anger":    {"emoji": "😡", "color": "#F94F4F", "bg": "#fdecea", "label": "Anger"},
    "fear":     {"emoji": "😨", "color": "#A855F7", "bg": "#f5f0ff", "label": "Fear"},
    "love":     {"emoji": "❤️",  "color": "#F97316", "bg": "#fff4ed", "label": "Love"},
    "surprise": {"emoji": "😲", "color": "#10B981", "bg": "#ecfdf5", "label": "Surprise"},
}

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

# ─── Prediction ───────────────────────────────────────────────────────────────
def predict(text: str):
    cleaned = preprocess(text)
    vec = tfidf.transform([cleaned])
    pred_idx = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    emotion = idx_to_emotion[pred_idx]
    return emotion, proba, cleaned

# ─── Keyword Extraction ───────────────────────────────────────────────────────
def get_keywords(cleaned: str, top_n: int = 10):
    vocab = tfidf.vocabulary_
    feature_names = tfidf.get_feature_names_out()
    vec = tfidf.transform([cleaned])
    coef = model.coef_  # shape (n_classes, n_features)
    pred_idx = model.predict(vec)[0]
    scores = coef[pred_idx] * vec.toarray()[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
    return keywords

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@700;800;900&display=swap');

/* ── Reset & Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    height: 100vh !important;
    overflow: hidden !important;
    background: #0f0f13 !important;
}
[data-testid="stAppViewContainer"] > section > div {
    height: 100vh;
    overflow: hidden;
}
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    height: 100vh;
    overflow: hidden;
}
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stButton button { display: none; }
header[data-testid="stHeader"], [data-testid="stToolbar"],
footer, #MainMenu { display: none !important; }

/* ── App Shell ── */
.app-shell {
    display: grid;
    grid-template-columns: 340px 1fr;
    grid-template-rows: 64px 1fr;
    height: 100vh;
    width: 100%;
    background: #0f0f13;
    overflow: hidden;
}

/* ── Top Bar ── */
.topbar {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    padding: 0 28px;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    gap: 12px;
}
.topbar-logo {
    font-family: 'Poppins', sans-serif;
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(135deg, #f9c74f 0%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}
.topbar-tagline {
    font-size: 13px;
    color: rgba(255,255,255,0.4);
    font-weight: 400;
    margin-left: 4px;
}
.topbar-badge {
    margin-left: auto;
    background: rgba(249,199,79,0.1);
    border: 1px solid rgba(249,199,79,0.25);
    color: #f9c74f;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* ── Left Panel ── */
.left-panel {
    background: rgba(255,255,255,0.025);
    border-right: 1px solid rgba(255,255,255,0.07);
    display: flex;
    flex-direction: column;
    padding: 24px 20px;
    gap: 16px;
    overflow: hidden;
}
.panel-label {
    font-size: 10px;
    font-weight: 700;
    color: rgba(255,255,255,0.3);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.text-area-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
textarea {
    width: 100% !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.85) !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    resize: none !important;
    padding: 14px !important;
    transition: border-color 0.2s;
    flex: 1;
}
textarea:focus {
    border-color: rgba(249,199,79,0.4) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(249,199,79,0.08) !important;
}
textarea::placeholder { color: rgba(255,255,255,0.2) !important; }

.analyze-btn {
    width: 100%;
    padding: 14px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #f9c74f 0%, #f97316 100%);
    color: #0f0f13;
    font-size: 15px;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    letter-spacing: 0.3px;
}
.analyze-btn:hover { opacity: 0.9; transform: translateY(-1px); }
.analyze-btn:active { transform: translateY(0); }

/* ── Right Panel ── */
.right-panel {
    display: grid;
    grid-template-rows: auto 1fr auto;
    gap: 0;
    overflow: hidden;
    padding: 20px 24px;
    background: #0f0f13;
}

/* ── Hero Result ── */
.hero-result {
    border-radius: 16px;
    padding: 22px 24px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 16px;
    transition: all 0.4s ease;
}
.hero-emoji {
    font-size: 56px;
    line-height: 1;
    filter: drop-shadow(0 4px 16px rgba(0,0,0,0.3));
}
.hero-text { flex: 1; }
.hero-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 4px;
}
.hero-emotion {
    font-family: 'Poppins', sans-serif;
    font-size: 36px;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero-confidence {
    font-size: 14px;
    font-weight: 500;
    opacity: 0.65;
    margin-top: 4px;
}

/* ── Probability Grid ── */
.prob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    overflow: hidden;
    margin-bottom: 16px;
}
.prob-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px 14px;
    transition: all 0.2s;
}
.prob-card.active {
    border-color: rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.06);
}
.prob-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.prob-name {
    font-size: 13px;
    font-weight: 600;
    color: rgba(255,255,255,0.75);
    display: flex;
    align-items: center;
    gap: 6px;
}
.prob-pct {
    font-size: 13px;
    font-weight: 700;
    color: rgba(255,255,255,0.55);
}
.prob-bar-bg {
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.5s ease;
}

/* ── Keywords ── */
.keywords-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}
.kw-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-right: 4px;
}
.kw-chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.6);
    border: 1px solid rgba(255,255,255,0.08);
    transition: background 0.2s;
}

/* ── Empty State ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: rgba(255,255,255,0.2);
    text-align: center;
    padding: 40px;
}
.empty-icon { font-size: 52px; margin-bottom: 8px; filter: grayscale(1) opacity(0.4); }
.empty-title { font-size: 18px; font-weight: 600; }
.empty-sub { font-size: 13px; line-height: 1.6; max-width: 280px; }

/* ── Accuracy pill ── */
.acc-pill {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.2);
    color: #10b981;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    margin-left: 8px;
}

/* ── Streamlit widget overrides ── */
[data-testid="stTextArea"] {
    flex: 1;
    display: flex;
    flex-direction: column;
}
[data-testid="stTextArea"] > div {
    flex: 1;
    display: flex;
    flex-direction: column;
}
[data-testid="stTextArea"] > div > div {
    flex: 1;
    display: flex;
    flex-direction: column;
}
[data-testid="stTextArea"] textarea {
    flex: 1;
    min-height: unset !important;
    height: 100% !important;
}
[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ─── App Shell: Top Bar ────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <span class="topbar-logo">🎭 Emotify</span>
  <span class="topbar-tagline">— Emotion Intelligence Engine</span>
  <span class="topbar-badge">NLP · Logistic Regression</span>
  <span class="acc-pill">✦ 86.15% Accuracy</span>
</div>
""", unsafe_allow_html=True)

# ─── Two-Column Layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([340, 1000], gap="small")

# ── LEFT PANEL ──
with left_col:
    st.markdown('<p class="panel-label">Input Text</p>', unsafe_allow_html=True)

    user_input = st.text_area(
        label="",
        placeholder="Type or paste any text here…\n\nExamples:\n• "I'm so happy today!"\n• "This makes me really angry"\n• "I feel so alone and sad"",
        height=320,
        key="text_input",
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("⚡ Analyze Emotion", use_container_width=True)

    st.markdown("""
    <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:14px; margin-top:4px;">
      <p class="panel-label">Detectable Emotions</p>
      <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">
        <span style="font-size:13px;">😄 Joy</span>
        <span style="font-size:13px;">😢 Sadness</span>
        <span style="font-size:13px;">😡 Anger</span>
        <span style="font-size:13px;">😨 Fear</span>
        <span style="font-size:13px;">❤️ Love</span>
        <span style="font-size:13px;">😲 Surprise</span>
      </div>
      <p style="font-size:11px; color:rgba(255,255,255,0.2); margin-top:14px; line-height:1.6;">
        Powered by TF-IDF vectorization + Logistic Regression trained on 16,000+ labeled samples.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT PANEL ──
with right_col:
    if analyze_clicked and user_input.strip():
        # Run prediction
        emotion, proba, cleaned = predict(user_input)
        keywords = get_keywords(cleaned)
        st.session_state.result = {
            "emotion": emotion,
            "proba": proba,
            "keywords": keywords,
        }
    elif analyze_clicked and not user_input.strip():
        st.session_state.result = "empty"

    res = st.session_state.result

    if res is None:
        # Empty state
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🎭</div>
          <div class="empty-title">No analysis yet</div>
          <div class="empty-sub">Enter any text on the left panel and click <strong>Analyze Emotion</strong> to get started.</div>
        </div>
        """, unsafe_allow_html=True)

    elif res == "empty":
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">✏️</div>
          <div class="empty-title">Nothing to analyze</div>
          <div class="empty-sub">Please type something in the text box before analyzing.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        emotion = res["emotion"]
        proba   = res["proba"]
        keywords = res["keywords"]
        meta    = EMOTION_META.get(emotion, EMOTION_META["joy"])
        confidence = proba[emotions_map[emotion]] * 100

        # ── Hero Result Card ──
        st.markdown(f"""
        <div class="hero-result" style="background: linear-gradient(135deg, {meta['color']}18 0%, {meta['color']}06 100%);
             border: 1px solid {meta['color']}35;">
          <div class="hero-emoji">{meta['emoji']}</div>
          <div class="hero-text">
            <div class="hero-label" style="color:{meta['color']};">Detected Emotion</div>
            <div class="hero-emotion" style="color:{meta['color']};">{meta['label']}</div>
            <div class="hero-confidence" style="color:{meta['color']};">
              Confidence: {confidence:.1f}%
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability Breakdown Grid ──
        st.markdown('<p class="panel-label" style="margin-bottom:8px;">Probability Breakdown</p>', unsafe_allow_html=True)

        prob_cards_html = '<div class="prob-grid">'
        for emo_name, emo_idx in sorted(emotions_map.items(), key=lambda x: -proba[x[1]]):
            m = EMOTION_META.get(emo_name, {"emoji": "🔵", "color": "#888", "label": emo_name.title()})
            pct = proba[emo_idx] * 100
            is_active = "active" if emo_name == emotion else ""
            bar_width = f"{pct:.1f}%"
            prob_cards_html += f"""
            <div class="prob-card {is_active}">
              <div class="prob-card-top">
                <span class="prob-name">{m['emoji']} {m['label']}</span>
                <span class="prob-pct" style="color:{m['color']};">{pct:.1f}%</span>
              </div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{bar_width}; background:{m['color']};"></div>
              </div>
            </div>"""
        prob_cards_html += "</div>"
        st.markdown(prob_cards_html, unsafe_allow_html=True)

        # ── Keywords ──
        if keywords:
            chips_html = "".join(
                f'<span class="kw-chip" style="border-color:{meta["color"]}30; color:{meta["color"]}99;">{kw}</span>'
                for kw in keywords
            )
            st.markdown(f"""
            <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:14px; margin-top:4px;">
              <div class="keywords-row">
                <span class="kw-label">Key Signals</span>
                {chips_html}
              </div>
            </div>
            """, unsafe_allow_html=True)
