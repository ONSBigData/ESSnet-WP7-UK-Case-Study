#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from collections import Counter

import matplotlib.pyplot as plt
import seaborn; seaborn.set()


# Read the data
sentiment = pd.read_csv("data/sentiment.csv", index_col=0, parse_dates = True)
comments = pd.read_csv("data/comments.csv", index_col=0, parse_dates = True)
posts = pd.read_csv("data/posts.csv", index_col=0, parse_dates = True)
emotions = pd.read_csv("data/emotions.csv", index_col=0, parse_dates = True)

posts = posts[~posts['tags'].isnull()] # 1,584



comments = comments.reset_index().merge(posts[['post_id', 'tags']], how='inner', on='post_id').set_index('created_time') # 365,395

tags = comments.tags.str.replace('[', '') \
        .str.replace(']', '') \
        .str.replace(' ', '') \
        .str.split(',')

tags = pd.get_dummies(tags.apply(pd.Series).stack()).sum(level=0)
tags.set_index(comments.index, inplace=True)

counts = tags.resample('H').sum()
summed = counts.sum(axis=0)
summed = summed[summed > 1000]
counts = counts[summed.index]
counts.fillna(0, inplace=True)


counts.to_csv('extra/counts.csv')



### 1. Background model ###
scores = counts.rolling(7*24).apply(lambda x: x[-1]/(x[0]+1))
# 1.2 Confirmation factor (n. of likes for the most authoritative comment)
likes = comments['like_count']
tags_likes = tags[summed.index].multiply(likes, axis=0)
weights = tags_likes.resample('H').max().fillna(0).rolling(7*24).apply(lambda x: max(x[0], x[-1]))
final_scores = scores.fillna(0) * weights.fillna(0)

# Extract trendings by hour
dStart = datetime(2017, 3, 6, 0 , 0)
dEnd = datetime(2017, 3, 31, 23, 59)
trending = final_scores.loc[dStart:dEnd].idxmax(axis=1)

# Adjust treshold based on topic?

### 2. Fit a distribution ###
