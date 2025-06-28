# -SENTIMENT-ANALYSIS-WITH-NLP

 *COMPANY*: CODTECH IT SOLUTIONS

 *NAME*: RAGHAV PANDEY
 
 *INTERN ID*: CT04DF122
 
 *DOMAIN*: MACHINE LEARNING
 
 *DURATION*: 4 WEEKS
 
 *MENTOR*: NEELA SANTOSH


 # Sentiment Analysis using TF-IDF and Logistic Regression

This project implements a **Sentiment Analysis** pipeline using **TF-IDF vectorization** and **Logistic Regression**. The goal is to classify customer reviews into positive or negative sentiments.

---

## 🧠 Overview

The pipeline includes:

- 📌 Text cleaning and preprocessing
- 🔤 Tokenization with stemming and stopword removal
- ✨ TF-IDF vectorization with unigram and bigram support
- ⚙️ Logistic Regression model for classification
- 📊 Evaluation using accuracy and confusion matrix

---

## 🗂️ Dataset

The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.

train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
test.tsv contains just phrases. You must assign a sentiment label to each phrase.
The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

---

## 🛠️ Preprocessing Steps

- Lowercasing
- Removing punctuation
- Tokenizing (using NLTK's `word_tokenize`)
- Removing English stopwords
- Stemming using `SnowballStemmer`

---

## 📚 Feature Extraction

Used **TfidfVectorizer** from `sklearn` with:

- `max_features=2300`
- `ngram_range=(1,2)`
- Custom tokenizer for tokenization, stemming, and stopword removal

---

## 🤖 Model

Used **Logistic Regression** with:
- `max_iter=1000` to ensure convergence
- `sklearn.linear_model.LogisticRegression`

---

## ✅ Evaluation

- Accuracy on test set
- Confusion matrix
- Optionally: Classification report

---
### libraries needed

pandas

numpy

nltk

scikit-learn

matplotlib



