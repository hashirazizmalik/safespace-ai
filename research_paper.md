# SafeSpace AI: Machine Learning-Based Mental Health Classification from Reddit Social Media Data

**Author:** Hashir Aziz Malik  
**Institution:** Institute of Management Sciences (IMSciences), Peshawar  
**Program:** BS Data Science  
**Email:** bsds.235301955@imsciences.edu.pk  
**GitHub:** github.com/hashirazizmalik/safespace-ai  
**Live Demo:** safespace-ai-nu.vercel.app  

---

## Abstract

Mental health disorders including depression, anxiety, and stress affect hundreds of millions of people globally, yet remain severely under-diagnosed due to stigma, limited access to professionals, and lack of awareness. This paper presents **SafeSpace AI**, a machine learning–powered web application that classifies user-written text into mental health categories (Depression, Anxiety, Stress, Normal) in real time. The system combines a TF-IDF text vectorizer (10,000 bigram features) with hand-crafted linguistic features including VADER sentiment scores, sentence count, word count, average word length, and first-person pronoun ratio. A Logistic Regression classifier trained on 33,000 Reddit posts from five subreddits (2019–2021) achieves a macro F1-score of **0.693** on a held-out test set. The model is deployed as a Flask web application on Vercel with a chat-style interface, interactive Plotly radar charts, and a confidence threshold mechanism that returns "Uncertain" for low-confidence predictions. The project demonstrates that classical NLP pipelines can produce deployable, interpretable mental health screening tools when deep learning infrastructure is unavailable.

**Keywords:** mental health classification, NLP, TF-IDF, logistic regression, VADER sentiment, Reddit, Flask, Vercel, SMOTE

---

## 1. Introduction

Mental health is one of the most pressing public health challenges of the 21st century. According to the World Health Organization (WHO), approximately 1 in 8 people worldwide live with a mental disorder, with depression and anxiety being the most prevalent. Despite this scale, the majority of affected individuals never receive formal diagnosis or treatment — a gap driven by social stigma, geographic inaccessibility, financial barriers, and a global shortage of mental health professionals.

Social media platforms, particularly Reddit, have emerged as spaces where individuals anonymously express mental health struggles with remarkable openness. Subreddits such as r/depression, r/anxiety, and r/stress receive thousands of posts daily, offering an unprecedented corpus of naturalistic mental health language. This creates an opportunity for automated, stigma-free, preliminary mental health screening using Natural Language Processing (NLP).

This paper presents **SafeSpace AI**, a system designed to:
1. Classify free-form user text into mental health categories in real time
2. Provide interpretable confidence scores and probability breakdowns
3. Operate as a publicly accessible web application with no sign-up required
4. Be honest about uncertainty through a confidence threshold mechanism

The system prioritizes **interpretability and deployability** over raw accuracy, making it practically accessible as a first-line awareness tool rather than a clinical diagnostic system.

---

## 2. Related Work

Early NLP approaches to mental health classification relied heavily on bag-of-words representations and classical classifiers. Coppersmith et al. (2014) used Twitter data with language model features to detect depression and PTSD, demonstrating that social media text carries significant diagnostic signal. Similarly, De Choudhury et al. (2013) predicted depression onset from Twitter posting behaviour with 70% accuracy.

More recent work has leveraged transformer-based architectures for improved accuracy. Amir et al. (2019) fine-tuned BERT on Reddit data and achieved over 80% F1 on depression classification. Ji et al. (2021) conducted a comprehensive survey of deep learning approaches to suicide and depression detection, noting that BERT and its variants consistently outperform classical models by 10–20% on benchmark datasets.

However, transformer models are computationally expensive at inference time, making serverless deployment challenging. Several studies have demonstrated that classical models with rich feature engineering can remain competitive in constrained deployment environments (Preoțiuc-Pietro et al., 2015). This work follows that tradition, building a practical, deployable system using Logistic Regression on a hybrid feature representation.

---

## 3. Dataset

### 3.1 Data Collection

Data was scraped from five Reddit subreddits spanning January 2019 to December 2021:

| Subreddit | Label Assigned | Posts Collected |
|---|---|---|
| r/depression | Depression | 11,000 |
| r/anxiety | Anxiety | 11,000 |
| r/stress | Stress | 5,500 |
| r/mentalhealth | Stress | 5,500 |
| r/AskReddit | Normal | 0* |

*No AskReddit CSV files were found in the raw data; the Normal class is handled by rule-based pre-filtering at inference time.

Each post contains: `title`, `selftext` (body), `subreddit`, `author`, `created_utc`, `score`.

**Total dataset size:** 33,000 posts (balanced at 11,000 per class after sampling)

### 3.2 Data Cleaning

The raw dataset contained 1,851,580 rows from 219 CSV files. The following cleaning steps were applied:

1. **Corruption removal:** 2,772 rows had subreddit names longer than 25 characters (column misalignment artefact); these were dropped.
2. **Class normalization:** Subreddit names lowercased and stripped; filtered to the five target communities.
3. **Downsampling:** Maximum 11,000 posts per subreddit to produce a balanced corpus.
4. **Text combination:** Post `title` and `selftext` concatenated to form a single input field.

### 3.3 Data Split

An 80/20 stratified train-test split was applied:
- Training set: 26,400 samples
- Test set: 6,600 samples

---

## 4. Methodology

### 4.1 Text Preprocessing

All text underwent the following NLP pipeline:

1. **Noise removal:** URLs, HTML tags, Reddit markdown (`[deleted]`, `[removed]`), and non-alphabetic characters stripped using regex.
2. **Tokenization:** NLTK word tokenizer applied after lowercasing.
3. **Stopword removal:** English stopwords removed (NLTK corpus).
4. **Lemmatization:** WordNetLemmatizer applied to reduce tokens to base forms.

```
Raw:   "I can't sleep anymore. Feeling hopeless... https://t.co/xyz"
Clean: "sleep anymore feel hopeless"
```

### 4.2 Feature Engineering

The feature matrix combines two components:

#### TF-IDF Text Features (10,000 dimensions)
A TF-IDF vectorizer with the following configuration:
- `max_features = 10,000`
- `ngram_range = (1, 2)` — unigrams and bigrams
- Fitted on cleaned, lemmatized text

Bigrams capture multi-word expressions such as *"panic attack"*, *"feel hopeless"*, *"can't sleep"* that carry stronger diagnostic signal than individual words.

#### Linguistic / Sentiment Features (5 dimensions)
Extracted from the raw (uncleaned) text to preserve emotional content:

| Feature | Description | Rationale |
|---|---|---|
| `vader_compound` | VADER compound sentiment score [-1, 1] | Negative sentiment correlates strongly with depression |
| `sentence_count` | Number of sentences | Stress posts tend to be longer/more elaborate |
| `word_count` | Total word count | Proxy for emotional engagement |
| `avg_word_length` | Mean character length per word | Cognitive load indicator |
| `pronoun_ratio` | (I + me + my + mine + myself) / word count | First-person focus correlates with depression |

**Combined feature matrix shape:** (33,000 × 10,005)

### 4.3 Class Imbalance Handling

The three-class training set (after excluding the absent Normal class) was already balanced at 8,800 samples per class after the 80/20 split. SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training set as an additional safeguard, maintaining the balanced distribution.

### 4.4 Model Selection

Three classifiers were evaluated on the same feature representation:

| Model | Macro F1 |
|---|---|
| Logistic Regression (baseline) | **0.699** |
| Support Vector Machine (LinearSVC) | ~0.68* |
| Random Forest (100 trees) | ~0.65* |

*Exact SVM/RF scores from the model comparison notebook cell.

**Logistic Regression was selected** as the deployment model for the following reasons:
1. Highest macro F1 on the test set
2. Native `predict_proba` support — required for the confidence threshold mechanism
3. Fast inference — suitable for serverless cold starts
4. Inherently interpretable — feature weights are inspectable

The LR model was trained with `max_iter=1000`, `C=1.0` (L2 regularization), and the `lbfgs` solver.

### 4.5 Confidence Threshold (Post-Processing)

A rule-based confidence filter was applied at inference time:

```
if max(predict_proba) < 0.55:
    return "Uncertain"
else:
    return argmax(predict_proba)
```

This prevents the model from confidently assigning a wrong category when the probability mass is spread across classes — a common failure mode in overlapping mental health language.

Additionally, two rule-based shortcuts handle edge cases without invoking the ML model:
- **Single neutral word** (e.g., "hi") → `Neutral greeting / State`
- **Short, strongly positive text** (VADER compound ≥ 0.6, ≤ 3 words, no distress keywords) → `Positive / Joy`

### 4.6 System Architecture

```
User Input (text)
       │
       ▼
Rule-based Pre-filter
 ├── Greeting? → "Neutral"
 ├── Positive? → "Positive / Joy"
 └── Otherwise ↓
       │
       ▼
 Text Cleaning → TF-IDF Transform
 Feature Extraction → VADER, linguistic features
       │
       ▼
 hstack([TF-IDF, numeric]) → LR.predict_proba()
       │
       ▼
 Confidence < 55%? → "Uncertain"
       │
       ▼
 Return: {category, confidence, probabilities}
       │
       ▼
 Flask JSON Response → Frontend Radar Chart
```

---

## 5. Results and Evaluation

### 5.1 Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Anxiety | 0.81 | 0.75 | 0.78 | 2,200 |
| Depression | 0.65 | 0.72 | 0.69 | 2,200 |
| Stress | 0.62 | 0.60 | 0.61 | 2,200 |
| **Macro Average** | **0.70** | **0.69** | **0.69** | **6,600** |

**Accuracy:** 69%  
**Macro F1:** 0.693

### 5.2 Key Observations

- **Anxiety** is the best-classified category (F1: 0.78). Anxiety language is lexically distinct — panic, worry, intrusive thoughts, physical symptoms — making it the most separable class.
- **Depression** achieves moderate performance (F1: 0.69). Depressive language heavily overlaps with anxiety (hopelessness, sleep disruption, social withdrawal).
- **Stress** is the weakest class (F1: 0.61). Stress posts are the most heterogeneous, ranging from work complaints to academic pressure to relationship issues — making them lexically diffuse.
- The model shows a consistent tendency to **confuse Depression and Stress**, reflecting a genuine semantic overlap in how people express these conditions on social media.

### 5.3 ROC-AUC Analysis

ROC-AUC scores (one-vs-rest):
- Anxiety: ~0.91
- Depression: ~0.84
- Stress: ~0.81

High AUC scores indicate the model has strong discriminative ability, even where F1 scores are moderate — the threshold tuning (0.55) is meaningful given this separation.

---

## 6. Deployment

### 6.1 Backend

The Flask application (`app.py`) exposes two endpoints:
- `GET /` — Serves the chat UI
- `POST /predict` — Accepts `{"text": "..."}`, returns `{"category": "...", "confidence": "...", "probabilities": {...}}`

Model files (`.joblib`) are loaded once at startup using absolute paths (`os.path.dirname(__file__)`) to ensure compatibility across local and serverless environments.

NLTK corpora are downloaded at startup to `/tmp/nltk_data` — the only writable directory in Vercel's serverless runtime.

### 6.2 Frontend

The chat interface (`templates/index.html`) is a single-page application using vanilla JavaScript and Plotly.js. Each prediction response renders:
1. A natural-language message with the predicted category and confidence
2. An interactive radar chart showing the probability distribution across all categories
3. Color-coded confidence badges per class

### 6.3 Infrastructure

| Component | Technology |
|---|---|
| Web Framework | Flask 3.1.3 |
| ML Runtime | scikit-learn 1.8.0, scipy 1.17.1 |
| NLP | NLTK 3.9.4 |
| Hosting | Vercel (serverless, Python runtime) |
| Version Control | GitHub (public repository) |
| Environment | Conda (Python 3.12) |

---

## 7. Discussion

### 7.1 Strengths

- **Deployable without GPU infrastructure** — The entire inference pipeline runs in a Vercel serverless function with sub-second latency.
- **Interpretable predictions** — Probability distributions and feature weights are inspectable; the system never acts as a black box.
- **Honest uncertainty handling** — The confidence threshold mechanism prevents confident misclassification, which is especially important in a mental health context.
- **Stigma-free interface** — The chat-style UI lowers the barrier to engagement compared to clinical questionnaires.

### 7.2 Limitations

- **No Normal class in training data** — The absence of AskReddit posts means the ML model cannot distinguish neutral text from clinical categories; this is compensated by rule-based pre-filtering but represents a structural gap.
- **Reddit-specific language distribution** — Models trained on Reddit posts may underperform on formal writing, spoken language, or non-English text.
- **Class overlap** — Depression and Stress share significant lexical overlap, which is a fundamental challenge in mental health NLP that classical models cannot fully resolve.
- **Not a clinical tool** — The system should not be used for diagnosis. It is an awareness and reflection tool only.
- **Static model** — The model does not update from user interactions, so it cannot adapt to language drift over time.

### 7.3 Ethical Considerations

Mental health AI carries significant ethical responsibility. This system includes a disclaimer on every prediction reminding users it is not a substitute for professional care. No user data is stored. The system is designed to encourage professional help-seeking, not replace it.

---

## 8. Conclusion and Future Work

This paper presented SafeSpace AI, a deployable mental health text classification system achieving 69.3% macro F1 on Reddit data. The system combines classical NLP with a confidence threshold mechanism and a chat-style interface to provide stigma-free, accessible mental health awareness.

**Future improvements in order of expected impact:**

1. **Fine-tuned transformer (DistilBERT/MentalBERT):** Expected +20–30% F1 improvement by capturing contextual semantics. Would require migration to a platform supporting larger model weights (e.g., Render, Railway).
2. **Normal class data:** Incorporating AskReddit or neutral news text into training would complete the four-class problem the app presents to users.
3. **Multilingual support:** Training on non-English Reddit data or using multilingual BERT to serve a global audience.
4. **Longitudinal tracking:** Allowing returning users to track emotional trends over time.
5. **Crisis detection:** A dedicated high-sensitivity classifier for suicide ideation with immediate resource referral.

---

## References

1. Coppersmith, G., Dredze, M., & Harman, C. (2014). Quantifying mental health signals in Twitter. *Proceedings of the ACL Workshop on Computational Linguistics and Clinical Psychology.*

2. De Choudhury, M., Gamon, M., Counts, S., & Horvitz, E. (2013). Predicting depression via social media. *ICWSM.*

3. Amir, S., Coppersmith, G., Carvalho, P., Silva, M. J., & Wallace, B. C. (2019). Mental health surveillance over social media. *arXiv preprint.*

4. Ji, S., Pan, S., Li, X., Cambria, E., Long, G., & Huang, Z. (2021). Suicidal ideation detection: A review of machine learning methods and applications. *IEEE Transactions on Computational Social Systems.*

5. Preoțiuc-Pietro, D., Sap, M., Schwartz, H. A., & Ungar, L. (2015). Mental illness detection at the World Well-Being Project. *CLPsych Workshop.*

6. Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM.*

7. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR, 16*, 321–357.

8. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.

---

*This paper is submitted as part of the BS Data Science program at the Institute of Management Sciences (IMSciences), Peshawar. The full codebase is publicly available at github.com/hashirazizmalik/safespace-ai.*
