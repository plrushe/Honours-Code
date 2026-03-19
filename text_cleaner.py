import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

STOPWORDS = set(stopwords.words("english")) - {"not"}
LEMM = WordNetLemmatizer()
URL_RE = re.compile(r"http\S+|www\.\S+")


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = []
        for t in X:
            t = str(t).lower()
            t = URL_RE.sub(" ", t)
            t = t.translate(str.maketrans("", "", string.punctuation))
            tokens = nltk.word_tokenize(t)

            out = []
            for w in tokens:
                if w in STOPWORDS:
                    continue
                if not w.isalpha():
                    continue
                out.append(LEMM.lemmatize(w))

            cleaned.append(" ".join(out))

        return np.array(cleaned, dtype=object)
