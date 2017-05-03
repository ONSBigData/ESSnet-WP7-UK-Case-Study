#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from utils import Mongo, d_to_df, get_scores, paragraph_sentiment, get_nrc_emotions

from nltk import tokenize, sent_tokenize
from nltk.sentiment.vader import normalize

# Load all others dictionaries
bing = pd.read_csv("lexicons/bing.csv")
bing.drop_duplicates('word', inplace = True)

afinn = pd.read_csv("lexicons/afinn.csv")
afinn.drop_duplicates('word', inplace = True)

syuzhet = pd.read_csv("lexicons/syuzhet.csv")
syuzhet.drop_duplicates('word', inplace = True)

nrc = pd.read_csv("lexicons/nrc2.csv", header=0, names = [u'word', u'anger', u'anticipation', u'disgust', u'fear', u'joy',
       u'negative', u'positive', u'sadness', u'surprise', u'trust']) # use modified lexicon
nrc.drop_duplicates('word', inplace = True)
nrc['value'] = nrc['positive']-nrc['negative']


# Load data from Mongo
mongo = Mongo('facebook', 'comments')
docs = [doc for doc in mongo.collection.find()]
mongo.close()
mongo_ids = [doc.pop('_id', None) for doc in docs] # exclude mongo generated ids
docs = d_to_df(docs)
docs['created_time'] = pd.to_datetime(docs['created_time'],format="%Y-%m-%dT%H:%M:%S+0000")
docs.set_index('created_time', inplace = True)
docs.drop_duplicates(['message', 'user.name', 'post_id'], inplace=True)
docs['n_sents'] = docs.message.apply(lambda x: len(sent_tokenize(x)))
docs['n_words'] = docs.message.apply(lambda x: len(tokenize.word_tokenize(x)))
docs = docs[docs['n_sents'] != 0].copy()

mongo = Mongo('facebook', 'posts')
posts = [doc for doc in mongo.collection.find()]
mongo.close()
mongo_ids = [post.pop('_id', None) for post in posts] # exclude mongo generated ids
posts = d_to_df(posts)
posts['created_time'] = pd.to_datetime(posts['created_time'],format="%Y-%m-%dT%H:%M:%S+0000")
posts.set_index('created_time', inplace=True)

# Calculating post title and message sentiment
posts['article_title'].fillna('', inplace=True)
posts['article_title_sentiment'] = posts.article_title.apply(paragraph_sentiment)
posts['message_sentiment'] = posts.message.apply(paragraph_sentiment)


# Calculating sentiment
bing_scores = get_scores(docs['message'], bing)
afinn_scores = get_scores(docs['message'], afinn)
syuzhet_scores = get_scores(docs['message'], syuzhet)
nrc_scores = get_scores(docs['message'], nrc) # used version 2 of the nrc lexicon
vader_scores = docs.message.apply(paragraph_sentiment)
all_methods = pd.DataFrame({'bing': bing_scores,
              'afinn': afinn_scores,
              'syuzhet': syuzhet_scores,
              'nrc': nrc_scores},
              index=docs.index).div(docs.n_sents, axis='index')
all_methods = all_methods.apply(lambda x: map(normalize, x))
all_methods['vader'] = vader_scores
sentiment = all_methods.loc[(all_methods!=0).any(axis=1)]
comments = docs.loc[(all_methods!=0).any(axis=1)]

# Calculate NRC emotions
emotions = get_nrc_emotions(comments['message'], nrc)
emotions.set_index(comments.index, inplace = True)
emotions = emotions.div(emotions.sum(axis=1), axis='index')*100
emotions.fillna(0, inplace=True)

# Save the data
comments.to_csv("data/comments.csv", encoding='utf-8')
sentiment.to_csv("data/sentiment.csv")
posts.to_csv("data/posts.csv",encoding='utf-8')
emotions.to_csv("data/emotions.csv",encoding='utf-8')
