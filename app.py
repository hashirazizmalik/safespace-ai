print("STEP 1: The script has started running!")

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import re
from scipy.sparse import hstack
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import traceback

# 1. NLTK downloads — use /tmp on Vercel (read-only filesystem except /tmp)
NLTK_DATA_PATH = '/tmp/nltk_data'
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_PATH)

for corpus in ['vader_lexicon', 'punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(corpus, download_dir=NLTK_DATA_PATH, quiet=True)

# 2. Flask app
app = Flask(__name__)

# 3. Load Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Loading Models...")
try:
    model = joblib.load(os.path.join(BASE_DIR, 'deployment_models', 'mental_health_lr_model.joblib'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'deployment_models', 'tfidf_vectoriser.joblib'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR loading models: {e}")
    model, tfidf = None, None

# 4. NLP Tools
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 5. Distress keywords (used ONLY in filter logic, NOT passed to model)
DISTRESS_KEYWORDS = {
    'pain', 'hurt', 'suffer', 'suffering', 'empty', 'numb', 'hollow', 'void', 'alone',
    'lonely', 'loneliness', 'worthless', 'hopeless', 'helpless', 'tired', 'exhausted',
    'broken', 'lost', 'dark', 'darkness', 'despair', 'sorrow', 'grief', 'anguish',
    'cry', 'crying', 'tears', 'sad', 'sadness', 'depressed', 'depression',
    'anxiety', 'anxious', 'stress', 'stressed', 'overwhelmed', 'burden',
    'die', 'death', 'dying', 'suicidal', 'suicide', 'end it', 'give up',
    'cant go on', 'no reason', 'pointless', 'meaningless', 'invisible', 'disappear',
    'trapped', 'drowning', 'shattered', 'bleeding', 'scream', 'silent', 'silence'
}

def extract_live_features(text):
    """
    Builds exactly 5 numeric features — must match training dimensions:
    [vader_compound, sentence_count, word_count, avg_word_len, pronoun_ratio]
    distress_density is intentionally excluded (model was NOT trained on it).
    """
    sentiment_dict = sia.polarity_scores(text)
    compound_score = sentiment_dict['compound']

    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    sentence_count = max(len(sentences), 1)
    word_count = max(len(words), 1)
    avg_word_length = sum(len(w) for w in words) / word_count

    pronouns = re.findall(r'\b(i|me|my|mine|myself)\b', text.lower())
    pronoun_ratio = len(pronouns) / word_count

    # ✅ Exactly 5 features — matches what the model was trained on
    numeric_features = np.array([[
        compound_score,
        sentence_count,
        word_count,
        avg_word_length,
        pronoun_ratio
    ]])

    # Clean text for TF-IDF
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    clean_tokens = [
        lemmatizer.lemmatize(w)
        for w in word_tokenize(clean_text)
        if w not in stop_words and len(w) > 1
    ]
    final_clean_text = " ".join(clean_tokens)

    text_vector = tfidf.transform([final_clean_text])
    return hstack([text_vector, numeric_features])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tfidf is None:
        return jsonify({"error": "Models failed to load. Check server logs."}), 500

    data = request.json
    user_message = data.get("text", "").strip()

    if not user_message:
        return jsonify({"error": "No text provided"}), 400

    try:
        lower_text = user_message.lower()
        word_count = len(user_message.split())
        sentiment = sia.polarity_scores(user_message)
        compound_score = sentiment['compound']
        has_distress = any(kw in lower_text for kw in DISTRESS_KEYWORDS)

        # Rule 1: Pure single-word neutral greetings only ("hi", "hello", "hey")
        # Tightened: word_count == 1, not just <= 4, to avoid catching "I feel okay"
        if word_count == 1 and compound_score == 0.0 and not has_distress:
            return jsonify({
                "category": "Neutral greeting / State",
                "confidence": "95.0%",
                "probabilities": {"Anxiety": 1.7, "Depression": 1.7, "Normal": 95.0, "Stress": 1.6}
            })

        # Rule 2: Strongly positive AND very short AND zero distress vocabulary
        # e.g. "great!", "I'm happy" — not poems or mixed-emotion text
        if word_count <= 3 and compound_score >= 0.6 and not has_distress:
            return jsonify({
                "category": "Positive / Joy",
                "confidence": "95.0%",
                "probabilities": {"Anxiety": 1.7, "Depression": 1.7, "Normal": 95.0, "Stress": 1.6}
            })

        # Rule 3: General positive statements without distress words
        # Catch sentences like "I love iqbals poetry"
        if compound_score >= 0.4 and not has_distress:
            return jsonify({
                "category": "Positive / Joy",
                "confidence": "85.0%",
                "probabilities": {"Anxiety": 5.0, "Depression": 5.0, "Normal": 85.0, "Stress": 5.0}
            })

        # Rule 4: Neutral normal statements without distress words
        if -0.1 <= compound_score < 0.4 and not has_distress and word_count <= 10:
            return jsonify({
                "category": "Normal",
                "confidence": "75.0%",
                "probabilities": {"Anxiety": 8.0, "Depression": 8.0, "Normal": 75.0, "Stress": 9.0}
            })

        # Everything else — ML model decides
        features = extract_live_features(user_message)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100

        # Confidence threshold — avoid committing to a wrong category
        if confidence < 55.0:
            return jsonify({
                "category": "Uncertain",
                "confidence": f"{confidence:.1f}%",
                "probabilities": {
                    cls: round(float(prob) * 100, 1)
                    for cls, prob in zip(model.classes_, probabilities)
                }
            })

        return jsonify({
            "category": str(prediction),
            "confidence": f"{confidence:.1f}%",
            "probabilities": {
                cls: round(float(prob) * 100, 1)
                for cls, prob in zip(model.classes_, probabilities)
            }
        })

    except Exception as e:
        # Print full traceback to terminal so you can debug
        print(f"\n--- PREDICTION ERROR ---")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting SafeSpace AI Server...")
    print("Visit: http://127.0.0.1:5001")
    app.run(debug=True, port=5001)