# Imports for cleaning operations
import time
import json
import csv
import pandas as pd
import string

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def preprocessor(text):
    text = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = [word.lower() for word in tokens]

    stopword_reference = stopwords.words('english')
    tokens = [tok for tok in tokens if (tok not in stopword_reference and len(tok) >= 3)]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    tagged_doc = pos_tag(tokens)

    noun_tags = ['NN','NNP','NNPS','NNS']
    verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']
    lemmatizer = WordNetLemmatizer()

    def lemmatize_helper(token, pos_tag):
        if pos_tag in noun_tags:
            return lemmatizer.lemmatize(token,'n')
        elif pos_tag in verb_tags:
            return lemmatizer.lemmatize(token, 'v')
        else:
            return lemmatizer.lemmatize(token,'n')

    pre_proc_text =   " ".join([lemmatize_helper(token,tag) for token,tag in tagged_doc])

    return pre_proc_text


def preprocess_export(infile, outfile):
    reader = pd.read_csv(infile, chunksize=50)
    chunk_count = 1
    for chunk in reader:
        print('Processing chunk: ('+ str(chunk_count) + ')')

        result = chunk['all_reviews'].apply(preprocessor)
        df = pd.concat([result, chunk['stars']], axis=1)
        df.to_csv(outfile, index=False, header=False, mode='a')
        chunk_count += 1


preprocess_export('data/yelp_reviews_clean.csv', 'data/dataset.csv')