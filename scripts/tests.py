import json
import pandas as pd
from fb_api import Mongo
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt
import numpy as np
import seaborn; seaborn.set()


def d_to_df(d):
    sanitized = json.loads(json.dumps(d))
    normalized = json_normalize(sanitized)
    df = pd.DataFrame(normalized)
    return df

mongo = Mongo('facebook', 'comments')
docs = [doc for doc in mongo.collection.find()]
mongo.close()
mongo_ids = [doc.pop('_id', None) for doc in docs] # exclude mongo generated ids


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
NUM_WEEKS = 3
axes = [plt.subplot(1,NUM_WEEKS,i) for i in range(1, NUM_WEEKS + 1)]
idx = [0, 168, 336, 492] # need to fix this with 7 days range
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
