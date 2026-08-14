# 🎭 Emotify - Emotion Predictor

Emotify is an interactive Natural Language Processing (NLP) web application that detects the underlying emotions in text in real-time. Built using Python, Scikit-Learn, and Streamlit, the app processes text inputs and maps them to six core emotion categories: **Joy**, **Sadness**, **Anger**, **Fear**, **Love**, and **Surprise**.

---

## 🌟 Features

*   **Premium Interactive UI**: Clean glassmorphism styling, custom typography, animations, and color-coded emotion displays.
*   **High-Accuracy Model**: Employs a **Logistic Regression** model achieving **86.15%** accuracy on the dataset.
*   **Probability Breakdown**: Visualizes probability distributions for each emotion class with progress bars.
*   **Keyword Detection**: Highlights key vocabulary words from the text input that contributed to the classification.
*   **Real-Time Predictions**: Highly optimized text preprocessing and inference workflow.

---

## 📂 Project Structure

```text
d:/python/NLP/emotionspredictor/
├── model.ipynb           # Jupyter notebook containing training code and analysis
├── emotions_predicter.py # Streamlit application file (Frontend)
├── emotions.txt          # Raw dataset containing text-label emotion pairs
├── lmodel.pkl            # Trained Logistic Regression classifier (Pickled)
├── tfidf.pkl             # Fitted TF-IDF Vectorizer (Pickled)
└── emotions_map.pkl      # Dictionary mapping emotion names to integer codes (Pickled)
```

---

## 📊 Model Performance

*   **Model Type**: Logistic Regression + TF-IDF Vectorization
*   **Accuracy**: `86.15%`
*   This model uses Term Frequency-Inverse Document Frequency (TF-IDF) features to give less weight to common, uninformative words and prioritize key emotion signals.

---

## 🛠️ Installation & Setup

1.  **Clone the workspace** or open the project folder `d:/python/NLP/emotionspredictor/` in your terminal.
2.  **Install dependencies**:
    ```bash
    pip install streamlit pandas numpy scikit-learn nltk
    ```

---

## 🚀 How to Run

### 1. Training the Model (Jupyter Notebook)
To retrain or export the models, open and run the cells in `model.ipynb`. The final cell contains the export script:
```python
import pickle

# Export Logistic Regression model, TF-IDF vectorizer, and emotions map
with open('lmodel.pkl', 'wb') as f:
    pickle.dump(lmodel, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('emotions_map.pkl', 'wb') as f:
    pickle.dump(emotions_num, f)
```

### 2. Launching the Streamlit App
To start the interactive frontend, execute the following command:
```bash
streamlit run emotions_predicter.py
```
This will spin up a local server and open the application in your default browser at `http://localhost:8501/`.

---

## 🧠 Preprocessing Pipeline

The application processes text inputs through a unified pipeline identical to the notebook training stage:
1.  **Lowercasing**: Standardizes all characters to lowercase.
2.  **Punctuation Removal**: Strips grammatical symbols and special characters.
3.  **Digit Removal**: Removes numbers and numerical characters.
4.  **Emoji/Non-ASCII Filtering**: Strips out non-ASCII character sequences.
5.  **Tokenization & Stopword Filtering**: Splits the text into words and filters out standard English stopwords (using `NLTK`).
