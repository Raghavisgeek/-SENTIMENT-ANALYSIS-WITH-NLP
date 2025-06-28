
# Sentiment Analysis using TF-IDF and Logistic Regression

This repository contains a complete implementation of a **Sentiment Analysis** pipeline using **TF-IDF vectorization** and **Logistic Regression**. The project is built entirely in Python using Jupyter Notebook and relies on several NLP preprocessing steps to clean and convert raw textual data into meaningful numerical features for classification.

## 🧠 Project Overview

Sentiment Analysis, also known as opinion mining, is a Natural Language Processing (NLP) technique used to determine whether textual data expresses a positive, negative, or neutral sentiment. This project aims to classify customer reviews or phrases into positive or negative sentiments using machine learning. Specifically, the model used here is **Logistic Regression**, trained on **TF-IDF (Term Frequency–Inverse Document Frequency)** features extracted from preprocessed text.

## 📦 Dataset

The dataset used for this project consists of a column of customer reviews labeled as "Phrase". The objective is to analyze the sentiment conveyed in these phrases and predict whether they reflect a positive or negative sentiment. The dataset is preprocessed and split into training and testing sets for model evaluation.

## 🧹 Text Preprocessing

Text preprocessing is a crucial step in the pipeline and includes:
- Lowercasing the text to maintain uniformity.
- Tokenizing using NLTK's `word_tokenize()` function.
- Removing punctuation and non-alphabetic characters.
- Removing English stopwords using NLTK’s stopword list.
- Applying stemming using `SnowballStemmer` to normalize word forms.

A custom `tokenize()` function encapsulates all these steps. This tokenizer is passed to `TfidfVectorizer` to ensure that feature extraction is aligned with the preprocessing logic.

## ✨ Feature Extraction: TF-IDF

TF-IDF is used to convert the cleaned and tokenized text into numerical vectors. We configure `TfidfVectorizer` with:
- `tokenizer=tokenize` (custom preprocessing)
- `ngram_range=(1, 2)` for both unigrams and bigrams
- `max_features=2300` to limit dimensionality

TF-IDF helps capture important terms that appear frequently in a document but rarely across documents, making it useful for classification tasks.

## 🔧 Model Training: Logistic Regression

Logistic Regression is used as the classifier. To ensure convergence during training, we set `max_iter=1000`. This model is trained on the TF-IDF vectors from the training set and evaluated on the test set using accuracy, precision, and confusion matrix.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

## 📊 Evaluation

The model's performance is measured using:
- **Accuracy**: Percentage of correct predictions.
- **Confusion Matrix**: To see true vs predicted classifications.
- **Classification Report** (optional): Shows precision, recall, F1-score.

## 📈 Visualization

Matplotlib is optionally used to visualize results such as confusion matrices or accuracy curves over iterations.

## 🗂 Dependencies

Required Python libraries:
- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `matplotlib`

Install them with:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Clone the repo or download the notebook.
2. Install required libraries.
3. Run the notebook from top to bottom.
4. View evaluation metrics and test results.

## ✅ Conclusion

This project is a foundational implementation of text classification using traditional NLP and ML techniques. It demonstrates how to effectively use TF-IDF and Logistic Regression with custom preprocessing to achieve reliable sentiment analysis.

## 👤 Author

Raghav Pandey 

---
