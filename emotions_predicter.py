import streamlit as st
import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Emotify - Emotion Predictor",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium UI/UX
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Hero Header Styling */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        background: linear-gradient(135deg, #FF3366, #FF9933);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #888888;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Cards and Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Emotion Cards styling */
    .emotion-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* Custom progress bar wrapper */
    .prob-container {
        margin-bottom: 0.8rem;
    }
    
    .prob-label {
        font-size: 0.95rem;
        display: flex;
        justify-content: space-between;
        margin-bottom: 3px;
    }

    /* Quote styling */
    .emotion-quote {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 1rem;
        color: #f1f1f1;
        border-left: 3px solid #ff3366;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Preprocessing Functions (same as train_model.py)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))

def preprocess_text(txt):
    if not isinstance(txt, str):
        return ""
    # Lowercase
    txt = txt.lower()
    # Remove punctuation
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    txt = ''.join([i for i in txt if not i.isdigit()])
    # Remove emojis/non-ascii
    txt = ''.join([i for i in txt if i.isascii()])
    # Tokenize and remove stopwords
    try:
        words = word_tokenize(txt)
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        words = word_tokenize(txt)
    cleaned_text = [i for i in words if i not in stop_words]
    return ' '.join(cleaned_text)

# Cache resource loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    try:
        lr_model_path = os.path.join(BASE_DIR, 'lmodel.pkl')
        tfidf_path = os.path.join(BASE_DIR, 'tfidf.pkl')
        map_path = os.path.join(BASE_DIR, 'emotions_map.pkl')
        
        with open(lr_model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(map_path, 'rb') as f:
            emotions_map = pickle.load(f)
            
        model_type = "Logistic Regression (Accuracy: 86.2%)"
        return model, vectorizer, emotions_map, model_type
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first by running the cells in `model.ipynb`.")
        return None, None, None, None

# Load model, vectorizer, map and model type
model, vectorizer, emotions_map, model_type = load_model()

# Header Section
st.markdown('<div class="hero-title">Emotify</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Discover the underlying emotions in your writing using machine learning</div>', unsafe_allow_html=True)

if model_type:
    st.markdown(f"<p style='text-align: center; color: #aaaaaa; font-size: 0.95rem; margin-top: -1.5rem; margin-bottom: 2rem;'>🔮 Active Model: <strong>{model_type}</strong></p>", unsafe_allow_html=True)

if model is not None:
    # Reverse the map to get emotion names from indices
    rev_emotions_map = {v: k for k, v in emotions_map.items()}
    
    # Emotion styling configs
    emotion_details = {
        'sadness': {'color': 'linear-gradient(135deg, #1e3c72, #2a5298)', 'emoji': '😢', 'quote': '"Heavy hearts, like heavy clouds in the sky, are best relieved by the letting of a little water." — Antoine de Saint-Exupéry'},
        'anger': {'color': 'linear-gradient(135deg, #e52d27, #b31217)', 'emoji': '😡', 'quote': '"For every minute you remain angry, you give up sixty seconds of peace of mind." — Ralph Waldo Emerson'},
        'love': {'color': 'linear-gradient(135deg, #ff007f, #ff5252)', 'emoji': '❤️', 'quote': '"There is only one happiness in this life, to love and be loved." — George Sand'},
        'surprise': {'color': 'linear-gradient(135deg, #f12711, #f5af19)', 'emoji': '😮', 'quote': '"Expect the unexpected, and whenever possible, be the unexpected." — Lynda Barry'},
        'fear': {'color': 'linear-gradient(135deg, #6441a5, #2a0845)', 'emoji': '😨', 'quote': '"Fear cuts deeper than swords." — George R.R. Martin'},
        'joy': {'color': 'linear-gradient(135deg, #11998e, #38ef7d)', 'emoji': '😊', 'quote': '"Joy is not in things; it is in us." — Richard Wagner'}
    }

    # Text Input Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Express your thoughts")
    user_input = st.text_area(
        "", 
        placeholder="Type or paste what's on your mind...", 
        height=150, 
        key="user_text"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("✨ Analyze Emotion", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Result Section
    if analyze_btn and user_input.strip() != "":
        # Preprocess input
        processed_input = preprocess_text(user_input)
        
        if processed_input.strip() == "":
            st.warning("Please enter some meaningful words to analyze.")
        else:
            # Transform and Predict
            vec_input = vectorizer.transform([processed_input])
            pred_idx = model.predict(vec_input)[0]
            pred_emotion = rev_emotions_map[pred_idx]
            
            # Predict Probabilities
            probs = model.predict_proba(vec_input)[0]
            
            # Visual result representation
            config = emotion_details.get(pred_emotion, {'color': '#555555', 'emoji': '💭', 'quote': ''})
            
            st.markdown(f"""
            <div class="glass-card" style="border-left: 5px solid {config['color'].split(',')[1].strip().replace(')', '')}">
                <h3 style='text-align: center; margin-bottom: 0.5rem;'>Analysis Result</h3>
                <div class="emotion-card" style="background: {config['color']}">
                    <span style="font-size: 3rem;">{config['emoji']}</span>
                    <h2 style="color: white; margin: 5px 0 0 0; font-weight: 800; font-size: 2rem;">{pred_emotion.upper()}</h2>
                </div>
                <div class="emotion-quote">{config['quote']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities breakdown Card
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Emotion Probabilities Breakdown")
            
            # Sort emotions by probability for nicer presentation
            sorted_indices = np.argsort(probs)[::-1]
            for idx in sorted_indices:
                emotion_name = rev_emotions_map[idx]
                prob = probs[idx]
                emoji = emotion_details.get(emotion_name, {}).get('emoji', '💭')
                color_hex = config['color'].split(',')[1].strip().replace(')', '') if emotion_name == pred_emotion else '#cccccc'
                
                # HTML layout for progress bar
                st.markdown(f"""
                <div class="prob-container">
                    <div class="prob-label">
                        <span>{emoji} <strong>{emotion_name.capitalize()}</strong></span>
                        <span>{prob:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Debug/Explanation Card (Key words contributing)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Key Words Detected")
            words_in_input = processed_input.split()
            matched_words = [w for w in words_in_input if w in vectorizer.vocabulary_]
            
            if matched_words:
                st.write("These words in your text contributed to the classification:")
                badges_html = " ".join([f"<span style='background-color:rgba(255,255,255,0.15); color:white; padding: 4px 10px; border-radius: 12px; margin-right: 5px; display: inline-block; margin-bottom: 5px; font-weight:600;'>🔑 {w}</span>" for w in set(matched_words)])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.info("None of the words in the input were in the training vocabulary. Prediction is based on prior probabilities.")
            st.markdown('</div>', unsafe_allow_html=True)
            
    elif analyze_btn:
        st.warning("Please type some text before running the analysis!")
else:
    st.info("Waiting for the model to be generated and loaded...")
