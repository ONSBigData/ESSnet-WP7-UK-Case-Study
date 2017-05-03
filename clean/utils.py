#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from pymongo import MongoClient

from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Mongo(object):
    def __init__(self, MONGODB_DB, MONGODB_COLLECTION, MONGODB_SERVER='localhost', MONGODB_PORT=27017):
        self.connection = MongoClient(
                    MONGODB_SERVER,
                    MONGODB_PORT
        )
        self.db = self.connection[MONGODB_DB]
        self.collection = self.db[MONGODB_COLLECTION]
    def process_item(self, item):
        try:
            self.collection.insert_one(dict(item))
            print 'Item inserted'
        except:
            print 'Insert failed for object:\n%s' %item
    def close(self):
        self.connection.close()

def d_to_df(d):
    sanitized = json.loads(json.dumps(d))
    normalized = pd.io.json.json_normalize(sanitized)
    df = pd.DataFrame(normalized)
    return df

def get_scores(text, vocab):
    vectorizer = CountVectorizer(vocabulary=vocab['word'])
    vect = vectorizer.fit_transform(text)
    values = np.array(vocab['value'])
    scores  = vect.dot(values)
    return scores

def get_nrc_emotions(text, vocab, plot_dist=False, n_words=10):
    vectorizer = CountVectorizer(vocabulary=vocab['word'])
    vect = vectorizer.fit_transform(text)
    colnames = [u'anger', u'anticipation', u'disgust',
                u'fear', u'joy', u'sadness', u'surprise',
                u'trust']
    values = csr_matrix(vocab[colnames].values)
    scores = vect.dot(values)
    if plot_dist:
        for emotion in colnames:
            vocabulary = vocab[vocab[emotion] > 0]['word']
            out = vect.tocsc()[:,np.array(vocabulary.index)]
            freqs = pd.DataFrame({'value': np.array(out.sum(axis = 0)).reshape(-1,)},
                                 index=vocabulary)
            ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
            top_freqs = freqs.sort_values('value', ascending=False).iloc[:n_words]
            top_freqs.plot(ax=ax, xticks=range(n_words), legend=False)
            ax.set_xticklabels(top_freqs.index,  rotation=90)
            ax.set_title(emotion[0].upper()+emotion[1:])
            plt.show()
            print emotion
            print top_freqs
            print
    return pd.DataFrame(scores.todense(), columns=colnames)

def paragraph_sentiment(paragraph):
    sid = SentimentIntensityAnalyzer()
    score = 0.0
    if paragraph:
        sentences = tokenize.sent_tokenize(paragraph)
        for sentence in sentences:
            ss = sid.polarity_scores(sentence)
            score += ss['compound']
        numofsents = float(len(sentences))
        score = score / numofsents
    return score
