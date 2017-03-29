import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats
import itertools
import ast

df = pd.read_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-comments-t-sentiment-test.csv', encoding='utf-8', index_col=0)
df['message'] = df['message'].fillna(u'')
df['word_count'] = df['message'].apply(lambda x: len(tokenize.word_tokenize(x)))

df2 = pd.read_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-posts-sentiment-test.csv', encoding='utf-8', index_col=0)

df2[df2.article_title_score]
# < 6 > Correlation between the reactions to an article and the sentiment of the article title

article_title_score = df2.article_title_score.values
article_title_pos = df2.article_title_pos.values
article_title_neu = df2.article_title_neg.values
article_title_neu = df2.article_title_neu.values



df2.reactions[0:1].values[0]
reactions = ast.literal_eval(df2.reactions[0:1].values[0])
type(ast.literal_eval(df2.reactions[0:1].values[0]))

df2.reactions = df2.reactions.apply(lambda x: ast.literal_eval(x))

# Filter df by the ids that are in df2
# Actually don't have to filter, because the reactions and article sentiments are contained in the same dataframe

# comparing df2.reactions, df2.article_title_score

reactions = df2.reactions.values

# keys -> angry / haha / like / love / sad / thankful / total_count / wow
# 1 option -> convert unordered dict to ordered dict then plot
# 2 option -> do a load of ifs
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
love = []
like = []
total_count = []
angry = []
haha = []
sad = []
thankful = []
wow = []

A=[]

def f():
    # Step 1
    #       Convert unordered dictionary to ordered dictionary
    # Step 2
    #       Iterate over ordered dictionary to
    # Or just post to stackoverflow

    return love, like, total_count, angry, haha, sad, thankful, wow

df['message_score'], df['message_pos'], df['message_neg'], df['message_neu'] = zip(*df['message'].map(f))



for reaction in reactions:
    A.append(reaction.items())
    for key, values in reaction.iteritems():



for reaction, title_score in zip(reactions, article_title_score):
    for key, value in reaction.iteritems():
        if key == 'love':
            plt.scatter(title_score, value, label=key, c = 'b')
        if key == 'like':
            plt.scatter(title_score, value, label=key, c='c')
        if key == 'total_count':
            plt.scatter(title_score, value, label=key, c='y')
        if key == 'angry':
            plt.scatter(title_score, value, label=key, c='g')
        if key == 'haha':
            plt.scatter(title_score, value, label=key, c='k')
        if key == 'sad':
            plt.scatter(title_score, value, label=key, c='r')
        if key == 'wow':
            plt.scatter(title_score, value, label=key, c='w')
        # if key == 'thankful':
        #     plt.scatter(title_score, value, label=key, c='m')
plt.legend(labels = [i for i in reaction.keys() if i != 'thankful'])

#








#new_array = []
#for i,j in enumerate(article_comments_sentiment):
#    new_array.append( [article_message_sentiment[i] for k in ]

# Reactions contained in the df2 array ->
