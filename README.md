# SafeSpace AI — Mental Health Support Chatbot

A machine learning–powered web app that analyzes text input and detects signs of mental health conditions including **Anxiety**, **Depression**, and **Stress**. Built with Flask and a Logistic Regression classifier trained on real Reddit posts.

---

## Live Demo

Deployed on Vercel: [safespace-ai.vercel.app](https://safespace-ai.vercel.app)

---

## Features

- **Real-time mental health classification** — detects Anxiety, Depression, Stress, or Normal state from free-form text
- **Confidence scoring** — shows how confident the model is in its prediction
- **Emotion breakdown radar chart** — interactive Plotly visualization showing probability distribution across all categories
- **Hybrid feature pipeline** — combines TF-IDF text features with VADER sentiment analysis, word/sentence counts, and pronoun ratio
- **Rule-based pre-filtering** — handles greetings and strongly positive short inputs without sending them to the model
- **Dark-themed chat UI** — clean, stigma-free interface

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | Logistic Regression (scikit-learn) |
| Text Features | TF-IDF Vectorizer |
| Sentiment | NLTK VADER |
| NLP | NLTK (tokenization, lemmatization, stopwords) |
| Visualization | Plotly.js (radar chart) |
| Deployment | Vercel |

---

## Model Details

The classifier was trained on Reddit posts scraped from mental health subreddits across 2019–2021, covering communities related to depression, anxiety, stress, loneliness, and general mental health.

**Feature Engineering:**
- TF-IDF bag-of-words (lemmatized, stopwords removed)
- VADER compound sentiment score
- Sentence count
- Word count
- Average word length
- First-person pronoun ratio (I, me, my, mine, myself)

**Output Categories:**
- `Depression`
- `Anxiety`
- `Stress`
- `Normal`
- `Positive / Joy`
- `Neutral greeting / State`

---

## Project Structure

```
safespace-ai/
├── app.py                        # Flask backend & prediction logic
├── requirements.txt              # Python dependencies
├── vercel.json                   # Vercel deployment config
├── templates/
│   └── index.html                # Chat UI frontend
└── deployment_models/
    ├── mental_health_lr_model.joblib   # Trained LR classifier
    └── tfidf_vectoriser.joblib         # Fitted TF-IDF vectorizer
```

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/hashirazizmalik/safespace-ai.git
cd safespace-ai
```

**2. Create and activate the conda environment**
```bash
conda create -n ml_project python=3.12
conda activate ml_project
pip install -r requirements.txt
```

**3. Start the server**
```bash
python app.py
```

Visit `http://127.0.0.1:5001` in your browser.

---

## Disclaimer

SafeSpace AI is not a substitute for professional mental health care. It is an educational project and should not be used for clinical diagnosis or treatment. If you or someone you know is in crisis, please contact a licensed mental health professional or a crisis helpline.

---

## Author

**Hashir Aziz Malik**  
BS Data Science — Institute of Management Sciences (IMSciences)  
[github.com/hashirazizmalik](https://github.com/hashirazizmalik)
