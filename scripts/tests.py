import json
import pandas as pd
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt
import numpy as np
import seaborn; seaborn.set()
seaborn.set_style("whitegrid")

from nltk import sent_tokenize

from datetime import datetime


def d_to_df(d):
    sanitized = json.loads(json.dumps(d))
    normalized = json_normalize(sanitized)
    df = pd.DataFrame(normalized)
    return df


# Load all others dictionaries
bing = pd.read_csv("bing.csv")
bing.drop_duplicates('word', inplace = True)

afinn = pd.read_csv("afinn.csv")
afinn.drop_duplicates('word', inplace = True)

syuzhet = pd.read_csv("syuzhet.csv")
syuzhet.drop_duplicates('word', inplace = True)

nrc = pd.read_csv("nrc2.csv", header=0, names = [u'word', u'anger', u'anticipation', u'disgust', u'fear', u'joy',
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
docs = docs[docs['n_sents'] != 0].copy()
docs['n_sents'] = docs.message.apply(lambda x: len(sent_tokenize(x)))

mongo = Mongo('facebook', 'posts')
posts = [doc for doc in mongo.collection.find()]
mongo.close()
mongo_ids = [post.pop('_id', None) for post in posts] # exclude mongo generated ids
posts = d_to_df(posts)
posts['created_time'] = pd.to_datetime(posts['created_time'],format="%Y-%m-%dT%H:%M:%S+0000")
posts.set_index('created_time', inplace=True)


# Comments Volume per day/per hour
dStart = datetime.date(2017, 2, 27)
dEnd = datetime.date(2017, 3, 31)
xticks = pd.date_range(start=dStart, end=dEnd, freq='W-Mon').union([dEnd])

ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
docs.resample('D').count()['comment_id'].plot(ax=ax, legend=True, label = 'Comments Volume per Day', xlim=(dStart, dEnd), xticks=xticks.to_pydatetime())
docs.resample('H').count()['comment_id'].plot(ax=ax, legend=True, label = 'Comments Volume per Hour', xlim=(dStart, dEnd), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
ax.annotate("\"Here's the full video of \nDonald J. Trump's first \npresidential address.\"",
            xy=(200, 180), xycoords='figure pixels', fontsize=8)
ax.annotate("\"Noel Fielding and\nSandi Toksvig to host new\nGreat British Bake Off.\"",
            xy=(700, 160), xycoords='figure pixels', fontsize=8)
ax.annotate("\"This is what it looks\nlike when Donald J. Trump\ndoesn't want to shake hands\nwith a foreign leader.\"",
            xy=(850, 180), xycoords='figure pixels', fontsize=8)
plt.show()

# Posts Volume per day
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
posts.resample('D').count()['post_id'].plot(ax=ax, legend=True, style='o-', label = 'Posts Volume per Day', ylim=(0,70), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
plt.show()

# Posts by main category
posts.groupby('main_category').count().sort_values('categories')['categories'].plot(kind='barh')


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

# All comments sentiment
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
sentiment.resample('H').mean().rolling(24).mean().plot(ax=ax, xlim=(dStart, dEnd), ylim=(-0.1, 0.12), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
sentiment.corr()

# Parent comments only sentiment
parents = sentiment[comments['parent_id'].isnull()]
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
parents.resample('H').mean().rolling(24).mean().plot(ax=ax, xlim=(dStart, dEnd), ylim=(-0.1, 0.12), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
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
plt.show()

##     7 days ma
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
positives.resample('D').mean().rolling(7).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime(), legend=True, label = 'Positives')
abs(negatives).resample('D').mean().rolling(7).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime(), legend=True, label = 'Negatives')
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
plt.show()

# Reactions
reactions = posts[['reactions.angry','reactions.love', 'reactions.sad', 'reactions.haha', 'reactions.wow']].copy()
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
reactions.resample('D').mean().rolling(7).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime())
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
plt.show()

# Emotions
emotions = get_nrc_emotions(comments['message'])
emotions.set_index(comments.index, inplace = True)
emotions = emotions.div(emotions.sum(axis=1), axis='index')*100
emotions.fillna(0, inplace=True)
ax = plt.figure(figsize=(15,20), dpi=100).add_subplot(111)
emotions.resample('H').mean().rolling(24).mean().plot(ax=ax, xlim=(dStart, dEnd), xticks=xticks.to_pydatetime(), colormap='Paired')
ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks]);
ax.set_xticklabels([], minor=True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
          ncol=4, fancybox=True, shadow=True)
plt.show()









# Save to csv
docs.reset_index()[['created_time', u'comment_count',
 u'comment_id',
 u'like_count',
 u'message',
 u'parent_id',
 u'post_id',
 u'user.id',
 u'user.name']].to_csv('comments.csv', encoding='utf-8', index = False)

posts.reset_index().to_csv('posts.csv', encoding='utf-8', index = False)


p_comments = pd.concat([comments[sentiment['vader'] > 0], positives], axis=1)
p_comments = p_comments[['post_id', 'vader']].reset_index().merge(posts[['post_id', 'main_category']], on = 'post_id', how='left').set_index('created_time')
p_pivoted = p_comments[['main_category', 'vader']].pivot_table('vader',
                           index = p_comments.index,
                           columns = 'main_category', fill_value = 0)
p_pivoted.resample('H').mean().rolling(24).mean().plot(colormap='cubehelix')
p_pivoted[['commentisfree', 'film', 'media', 'politics', 'uk-news', 'us-news', 'world']].resample('H').mean().rolling(24).mean().plot()

n_comments = pd.concat([comments[sentiment['vader'] < 0], negatives], axis=1)
n_comments = n_comments[['post_id', 'vader']].reset_index().merge(posts[['post_id', 'main_category']], on = 'post_id', how='left').set_index('created_time')
n_pivoted = n_comments[['main_category', 'vader']].pivot_table('vader',
                           index = n_comments.index,
                           columns = 'main_category', fill_value = 0)
n_pivoted.abs().resample('H').mean().rolling(24).mean().plot()

n_pivoted[['commentisfree', 'politics', 'uk-news', 'us-news', 'world']].abs().resample('H').mean().rolling(24).mean().plot()





import time

start = time.time()
scores = get_nrc_scores(docs['message'])
end = time.time()
print(end - start) # 10 secs


docs['pos'] = scores[:,0]
docs['neg'] = scores[:,1]

docs[docs['parent_id'].isnull()][['pos', 'neg']].resample('D').mean().plot()


p = posts[posts['main_category'] == 'politics']['post_id']



emotions = get_nrc_emotions(docs['message'])
emotions.set_index(docs.index, inplace = True)
import datetime

fig = plt.figure(figsize=(10,20))
fig.suptitle('Politics Posts', fontsize=16, fontweight='bold')
n=5
datemin = datetime.date(2017, 2, 27)
datemax = datetime.date(2017, 3, 31)

axes = [plt.subplot(n,1,i) for i in range(1, n + 1)]
ax = axes[0]
pv=ax.plot(posts[posts['post_id'].isin(p)].resample('D').count()['post_id'])
ax.set_xlim(datemin, datemax)
ax.set_ylabel("Posts Volume")
ax = axes[1]
v=ax.plot(docs[docs['post_id'].isin(p)].resample('D').count()['pos'])
ax.set_xlim(datemin, datemax)
ax.set_ylabel("Comments Volume")
ax = axes[2]
pn=ax.plot(docs[docs['post_id'].isin(p)][['pos', 'neg']].apply(lambda x: map(normalize, x)).resample('D').mean())
ax.legend(pn, ["Pos", "Neg"])
ax.set_xlim(datemin, datemax)
ax.set_ylabel("NRC Sentiment")
ax = axes[3]
am=ax.plot(all_methods[docs['post_id'].isin(p)].apply(lambda x: map(normalize, x)).resample('D').mean())
ax.legend(am, [u'afinn', u'bing', u'nrc', u'syuzhet'], loc=0)
ax.set_xlim(datemin, datemax)
ax.set_ylabel("All overall Sentiment")
ax = axes[4]
e=ax.plot(emotions[docs['post_id'].isin(p)].resample('D').mean())
ax.legend(e, emotions.columns, loc=0)
ax.set_xlim(datemin, datemax)
ax.set_ylabel("NRC Emotions")


# Correlation between comments count and posts count (0.88)
pd.concat([posts[posts['post_id'].isin(p)].resample('D').count()['post_id'], docs[docs['post_id'].isin(p)].resample('D').count()['pos']], axis=1).corr()

posts.groupby('main_category').count()['categories'].plot(kind='barh')

docs.reset_index()[['created_time', u'comment_count',
 u'comment_id',
 u'like_count',
 u'message',
 u'parent_id',
 u'post_id',
 u'user.id',
 u'user.name']].to_csv('comments.csv', encoding='utf-8', index = False)

posts.reset_index().to_csv('posts.csv', encoding='utf-8', index = False)



# comments, sentiment

comments['vader'] = sentiment['vader']
comments['date_h'] =  comments.index.strftime("%Y-%m-%dT%H")

by_h = comments.group_by(['date_h','post_id']).count()

max_p = by_h.reset_index().group_by('date_h', level=0).agg({ 'Max_post' : lambda x: x,
                                         'Max_count': np.max,
                                         'Percentage': lambda x: np.max(x)/float(np.sum(x))})



comments.groupby('post_id').resample('H').count()


from collections import Counter

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
top10 = max_post.reset_index().sort_values('tot_count', ascending=False).head(10)

top10 = top10.merge(posts[['post_id', 'article_title', 'message']], left_on='max_post', right_on='post_id', how='left')

for rank, idx in enumerate(top10.index):
    msg = top10.loc[idx,'article_title']
    try:
        if np.isnan(msg):
            msg = top10.loc[idx,'message']
    except TypeError:
        pass
    print "Post n.%d with title '%s'" %(rank+1, msg)
