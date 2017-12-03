#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from collections import Counter

import matplotlib.pyplot as plt
import seaborn; seaborn.set()


# Read the data (data files have been produced by running script 2_get_sentiment.py)
sentiment = pd.read_csv("data/sentiment.csv", index_col=0, parse_dates = True)
comments = pd.read_csv("data/comments.csv", index_col=0, parse_dates = True)
posts = pd.read_csv("data/posts.csv", index_col=0, parse_dates = True)
emotions = pd.read_csv("data/emotions.csv", index_col=0, parse_dates = True)


dStart = datetime.date(2017, 2, 27)
dEnd = datetime.date(2017, 3, 31)
xticks = pd.date_range(start=dStart, end=dEnd, freq='W-Mon').union([dEnd])

# Posts Volume per day
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
posts.resample('D').count()['post_id'].plot(ax=ax, legend=True, label = 'Posts Volume per Day', ylim=(0,70), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
plt.show()

# Comments Volume per day/per hour
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
comments.resample('D').count()['comment_id'].plot(ax=ax, legend=True, label = 'Comments Volume per Day', xlim=(dStart, dEnd), xticks=xticks.to_pydatetime())
comments.resample('H').count()['comment_id'].plot(ax=ax, legend=True, label = 'Comments Volume per Hour', xlim=(dStart, dEnd), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
plt.show()

# Top 10 commented articles per hour
def most_common(s):
    try:
        mc = Counter(s).most_common(1)[0]
    except IndexError:
        mc = (None, 0)
    tot = float(len(s))
    if mc[1] != 0:
        perc = mc[1]/tot
    else:
        perc = 0
    return pd.Series({'max_post':mc[0],
                      'max_count':mc[1],
                      'tot_count': tot,
                      'perc': perc})

r = comments.resample('H')
max_post = r.post_id.apply(most_common).unstack(level=[1])
top10 = max_post.reset_index().sort_values('max_count', ascending=False).head(10)
top10 = top10.merge(posts[['post_id', 'article_title', 'message']],
                    left_on='max_post', right_on='post_id', how='left')

# Posts by main category
counts = posts.groupby('main_category').count()
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
counts.sort_values('categories')['categories'].plot(ax=ax, kind='barh')
ax.tick_params(axis='y', labelsize=12)
plt.show()


# All comments sentiment
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
sentiment.resample('H').mean().rolling(24).mean().plot(ax=ax, xlim=(dStart, dEnd), ylim=(-0.1, 0.12), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
ax.legend(loc=1, prop={'size':12})
sentiment.corr()

# Parent comments only sentiment
parents = sentiment[comments['parent_id'].isnull()]
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
parents.resample('H').mean().rolling(24).mean().plot(ax=ax, xlim=(dStart, dEnd), ylim=(-0.1, 0.12), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
ax.legend(loc=1, prop={'size':12})
plt.show()
parents.corr()


# Vader Pos vs. Neg
positives = sentiment['vader'][sentiment['vader'] > 0]
negatives = sentiment['vader'][sentiment['vader'] < 0]

##     24 hours ma
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
positives.resample('H').mean().rolling(24).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime(), legend=True, label = 'Positives')
abs(negatives).resample('H').mean().rolling(24).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime(), legend=True, label = 'Negatives')
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
ax.legend(loc=1, prop={'size':12})
plt.show()

##     7 days ma
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
positives.resample('D').mean().rolling(7).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime(), legend=True, label = 'Positives')
abs(negatives).resample('D').mean().rolling(7).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime(), legend=True, label = 'Negatives')
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
ax.legend(loc=2, prop={'size':12})
plt.show()

# Reactions
reactions = posts[['reactions.angry','reactions.love', 'reactions.sad', 'reactions.haha', 'reactions.wow']].copy()
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
reactions.resample('D').mean().rolling(7).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
ax.legend(loc=0, prop={'size':12})
plt.show()

# Emotions
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
emotions.resample('H').mean().rolling(24).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime(), colormap='Paired')
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=4, fancybox=True, shadow=True, prop={'size':12})
plt.show()

# Relationships

# Hexagonal Hexbin Plot - up to 100 word count
# Logarithmic Color Bar
xmin, xmax, ymin, ymax = 0, 100, -1, 1
x = np.array(comments[(comments['n_words'] > xmin) & (comments['n_words'] <= xmax)]['n_words'])
y = np.array(sentiment[(comments['n_words'] > xmin) & (comments['n_words'] <= xmax)]['vader'])
fig, ax = plt.subplots(ncols=1, figsize=(15,20), dpi=100)
hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='viridis', extent = [xmin, xmax, ymin, ymax])
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title('Hexagonal Bin Plot of Comment Sentiment vs Comment Word Count with Log colour scale'
             '\nfor Facebook Comments in response to Guardian News Articles'
             '\nMax Word Count constrained to 100', fontsize = 16)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(Counts)', fontsize = 15)
plt.show()

# Correlation between article title/message
valid_posts = posts[posts['article_title'] != ''].copy()

x = np.array(valid_posts['article_title_sentiment'])
y = np.array(valid_posts['message_sentiment'])

gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
fit = x * gradient + intercept

fig, ax = plt.subplots(ncols=1, figsize=(15,20), dpi=100)
plt.scatter(x, y,
            s = 5,
            label = 'Scatter Plot Title vs Description',
            c = 'b')
plt.plot(x, fit, label = 'Linear Regression Fit', c='c')
plt.title('Article Title Sentiment vs Article Description Sentiment'
          '\n$R^2$ = {} gradient = {}'.format(round(r_value ** 2, 4), round(gradient ** 2, 4)),
          fontsize = 18)
plt.legend(loc=2, prop={'size':12})
plt.xlabel('Article Title Sentiment', fontsize = 12)
plt.ylabel('Article Description Sentiment',fontsize = 12)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()


# Correlation between article title and facebook comments (grouped)
valid_posts.set_index('post_id', inplace=True)
all_comments = pd.concat([comments, sentiment['vader']], axis = 1)
all_comments = all_comments[['post_id', 'vader']].groupby('post_id').mean()
combined = pd.merge(valid_posts[['article_title', 'article_title_sentiment']], all_comments, left_index=True, right_index=True, how='inner')

x = np.array(combined['article_title_sentiment'])
y = np.array(combined['vader'])

gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
fit = x * gradient + intercept

fig, ax = plt.subplots(ncols=1, figsize=(15,20), dpi=100)
plt.scatter(x, y, s = 1, label = 'Scatter Plot Article Title vs Comments', c = 'b')
plt.plot(x, fit, label = 'Linear Regression Fit', c='c')
plt.title('Article Title Sentiment vs Comments Sentiment (Grouped) - Linear Regression Fit\n'
          '$R^2$ = {}'.format(round(r_value ** 2, 4)),
          fontsize = 18)
plt.legend(loc=2, prop={'size':12})
plt.xlabel('Article Title Sentiment', fontsize = 12)
plt.ylabel('Comments Sentiment Mean(Grouped)', fontsize = 12)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()


#  Correlation between parent and childs (grouped) comments
childs = pd.concat([comments[~comments['parent_id'].isnull()], sentiment[~comments['parent_id'].isnull()]['vader']], axis = 1)
childs = childs[['parent_id', 'vader']].groupby('parent_id').mean()
parents = pd.concat([comments[comments['parent_id'].isnull()], sentiment[comments['parent_id'].isnull()]['vader']], axis = 1)
parents.set_index('comment_id', inplace = True)
relatives = pd.merge(parents[['post_id', 'vader']], childs, left_index=True, right_index=True, how='inner', suffixes=('_parent', '_child'))

x = np.array(relatives['vader_parent'])
y = np.array(relatives['vader_child'])

gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
fit = x * gradient + intercept

fig, ax = plt.subplots(ncols=1, figsize=(15,20), dpi=100)
plt.scatter(x, y, s = 1, label = 'Scatter Plot Parent vs Child', c = 'b')
plt.plot(x, fit, label = 'Linear Regression Fit', c='c')
plt.title('Parent Sentiment vs Child Sentiment Mean (Grouped) - Linear Regression Fit\n'
          '$R^2$ = {}'.format(round(r_value ** 2, 4)),
          fontsize = 18)
plt.legend(loc=2, prop={'size':12})
plt.xlabel('Parent Sentiment', fontsize = 12)
plt.ylabel('Child Sentiment Mean(Grouped)', fontsize = 12)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.show()
