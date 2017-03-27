import json
import pandas as pd
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt
import numpy as np
import seaborn; seaborn.set()

from nltk import sent_tokenize


def d_to_df(d):
    sanitized = json.loads(json.dumps(d))
    normalized = json_normalize(sanitized)
    df = pd.DataFrame(normalized)
    return df

mongo = Mongo('facebook', 'comments')
docs = [doc for doc in mongo.collection.find()]
mongo.close()
mongo_ids = [doc.pop('_id', None) for doc in docs] # exclude mongo generated ids

mongo = Mongo('facebook', 'posts')
posts = [doc for doc in mongo.collection.find()]
mongo.close()
mongo_ids = [post.pop('_id', None) for post in posts] # exclude mongo generated ids



posts = d_to_df(posts)
docs = d_to_df(docs)
docs['created_time'] = pd.to_datetime(docs['created_time'],format="%Y-%m-%dT%H:%M:%S+0000")
docs.set_index('created_time', inplace = True)

# Comments Volume per day
docs.resample('D').count()['comment_id'].plot(legend=True, label = 'Comments Volume per Day')
docs.resample('H').count()['comment_id'].plot(legend=True, label = 'Comments Volume per Hour')




# Calendar matrix
docs['hour'] = [x.hour for x in docs.index]
docs['day'] = [x.date() for x in docs.index]

by_hw = docs.groupby(['day','hour']).count().reset_index()

fig = plt.figure(figsize=(20,10))
NUM_WEEKS = 4
axes = [plt.subplot(1,NUM_WEEKS,i) for i in range(1, NUM_WEEKS + 1)]
idx = [0, 168, 336, 492, 608] # need to fix this with 7 days range
for i in range(NUM_WEEKS):
    ax = axes[i]
    week = by_hw.iloc[idx[i]:idx[i+1]].copy()
    im = ax.imshow(np.array(pd.pivot_table(week, values='comment_id', index='hour', columns='day')),
            cmap = 'OrRd',
            label = 'Volume')
    fig.colorbar(im, ax=ax)
    ax.set_yticks(np.arange(0, 24, 1));
    ax.set_xticks(np.arange(0, 7, 1));
    ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    ax.grid(None)

docs['n_sents'] = docs.message.apply(lambda x: len(sent_tokenize(x)))


import time

start = time.time()
scores = get_nrc_scores(docs['message'])
end = time.time()
print(end - start) # 10 secs


docs['pos'] = scores[:,0]
docs['neg'] = scores[:,1]

docs[docs['parent_id'].isnull()][['pos', 'neg']].resample('D').mean().plot()


p = posts[posts['main_category'] == 'politics']['post_id']
posts['created_time'] = pd.to_datetime(posts['created_time'],format="%Y-%m-%dT%H:%M:%S+0000")
posts.set_index('created_time', inplace=True)

import datetime

fig = plt.figure(figsize=(10,20))
fig.suptitle('Politics Posts', fontsize=16, fontweight='bold')
n=3
datemin = datetime.date(2017, 2, 27)
datemax = datetime.date(2017, 3, 20)

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

posts.groupby('main_category').count()['categories'].plot(kind='barh')

docs.reset_index()[['created_time', u'comment_count',
 u'comment_id',
 u'like_count',
 u'message',
 u'parent_id',
 u'post_id',
 u'user.id',
 u'user.name']].to_csv('comments.csv', encoding='utf-8', index = False)

posts.to_csv('posts.csv', encoding='utf-8', index = False)
