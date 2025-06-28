
# Sentiment Analysis using TF-IDF and Logistic Regression

This repository contains a complete implementation of a **Sentiment Analysis** pipeline using **TF-IDF vectorization** and **Logistic Regression**. The project is built entirely in Python using Jupyter Notebook and relies on several NLP preprocessing steps to clean and convert raw textual data into meaningful numerical features for classification.

##  Project Overview

Sentiment Analysis, also known as opinion mining, is a Natural Language Processing (NLP) technique used to determine whether textual data expresses a positive, negative, or neutral sentiment. This project aims to classify customer reviews or phrases into positive or negative sentiments using machine learning. Specifically, the model used here is **Logistic Regression**, trained on **TF-IDF (Term Frequency–Inverse Document Frequency)** features extracted from preprocessed text.

## Dataset Description


The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.


train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
test.tsv contains just phrases. You must assign a sentiment label to each phrase.
The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

## Text Preprocessing

Text preprocessing is a crucial step in the pipeline and includes:
- Lowercasing the text to maintain uniformity.
- Tokenizing using NLTK's `word_tokenize()` function.
- Removing punctuation and non-alphabetic characters.
- Removing English stopwords using NLTK’s stopword list.
- Applying stemming using `SnowballStemmer` to normalize word forms.

A custom `tokenize()` function encapsulates all these steps. This tokenizer is passed to `TfidfVectorizer` to ensure that feature extraction is aligned with the preprocessing logic.

##  Feature Extraction: TF-IDF

TF-IDF is used to convert the cleaned and tokenized text into numerical vectors. We configure `TfidfVectorizer` with:
- `tokenizer=tokenize` (custom preprocessing)
- `ngram_range=(1, 2)` for both unigrams and bigrams
- `max_features=2300` to limit dimensionality

TF-IDF helps capture important terms that appear frequently in a document but rarely across documents, making it useful for classification tasks.

##  Model Training: Logistic Regression

Logistic Regression is used as the classifier. To ensure convergence during training, we set `max_iter=1000`. This model is trained on the TF-IDF vectors from the training set and evaluated on the test set using accuracy, precision, and confusion matrix.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

## Evaluation

The model's performance is measured using:
- **Accuracy**: Percentage of correct predictions.
- **Confusion Matrix**: To see true vs predicted classifications.
- **Classification Report** (optional): Shows precision, recall, F1-score.

##  Visualization

Matplotlib is optionally used to visualize results such as confusion matrices or accuracy curves over iterations.

##  Dependencies

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

##  How to Run

1. Clone the repo or download the notebook.
2. Install required libraries.
3. Run the notebook from top to bottom.
4. View evaluation metrics and test results.

##  Conclusion

This project is a foundational implementation of text classification using traditional NLP and ML techniques. It demonstrates how to effectively use TF-IDF and Logistic Regression with custom preprocessing to achieve reliable sentiment analysis.

## Author

Raghav Pandey 

---
