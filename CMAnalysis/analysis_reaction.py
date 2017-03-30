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

# Extract Article Title Sentiments
article_title_score = df2.article_title_score.values
article_title_pos = df2.article_title_pos.values
article_title_neu = df2.article_title_neg.values
article_title_neu = df2.article_title_neu.values

# Extract reactions from the retarded pandas unicode format
df2.reactions = df2.reactions.apply(lambda x: ast.literal_eval(x))
reactions = df2.reactions.values.tolist()
# reaction dataframe
rdf = pd.DataFrame(reactions)
# Merge DataFrames
# Drop index as missing some indices cuasing concat not to line everything up appropriate
df2.reset_index(drop=False, inplace=True)
df2 = pd.concat([df2, rdf], ignore_index=False, axis = 1)

# < 6 > Correlation between the reactions to an article and the sentiment of the article title

# Correlation plots between Article Title and Reactions



fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))
# ax = fig.add_subplot(111)
for column in df2.columns[21:]:
    ax1.scatter(df2.article_title_score,df2['{}'.format(column)], label = column)
for column in df2.columns[21:]:
    ax2.scatter(df2.article_title_pos,df2['{}'.format(column)], label = column)
for column in df2.columns[21:]:
    ax3.scatter(df2.article_title_neg,df2['{}'.format(column)], label = column)
for column in df2.columns[21:]:
    ax4.scatter(df2.article_title_neu,df2['{}'.format(column)], label = column)
ax1.set_xlabel('common xlabel')
ax1.set_ylabel('common ylabel')

plt.xlabel('asdasd')
plt.ylabel('sadasd')
plt.legend()

# Reactions All on Same Plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))
for axis, sentiment in zip(list(itertools.chain(*axes)), df2.columns[17:21]):
    print axis
    print sentiment
    #for ax in axis:
    for column in df2.columns[21:]:
        axis.scatter(df2['{}'.format(sentiment)],df2['{}'.format(column)], label = column, s = 4)
    axis.set_xlabel('Sentiment')
    axis.set_ylabel('Frequency')
    axis.set_title('Article Title Sentiment vs Reaction Frequency - {}'.format(sentiment))
plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
plt.suptitle('Article Title Sentiment vs Reaction Frequency')

# Reactions on Separate Plots
# Article Title Sentiment contained between df2.columns[17:21]
# Article Message Sentiment constained between df2.columns[13:17]


for reaction in df2.columns[21:]:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))
    # Article Message Sentiment:
    for axis, sentiment in zip(list(itertools.chain(*axes)), df2.columns[13:17]):
    # Article Title Sentiment:
    # for axis, sentiment in zip(list(itertools.chain(*axes)), df2.columns[17:21]):
        print axis
        print sentiment
        #for ax in axis:

        # axis.scatter(df2['{}'.format(sentiment)],df2['{}'.format(reaction)], label = reaction, s = 4)
        axis.hexbin(df2['{}'.format(sentiment)],df2['{}'.format(reaction)], gridsize=25, bins='log')
        axis.set_xlabel('{} Sentiment'.format(sentiment))
        axis.set_ylabel('Frequency')
        axis.set_title('{} vs {} Frequency '.format(sentiment, reaction))
        axis.set_xlim([0,1])
        axis.set_ylim([0, 15000])
    list(itertools.chain(*axes))[0].set_xlim([-1,1])
    plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    # Title Sentiment
    # plt.suptitle('Article Title Sentiment vs {} Frequency'.format(reaction))
    # Message Sentiment
    plt.suptitle('Article Message Sentiment vs {} Frequency'.format(reaction))





df2.reactions[0:1].values[0]
reactions = ast.literal_eval(df2.reactions[0:1].values[0])
type(ast.literal_eval(df2.reactions[0:1].values[0]))

df2.reactions = df2.reactions.apply(lambda x: ast.literal_eval(x))
reactions = df2.reactions.values

# Filter df by the ids that are in df2
# Actually don't have to filter, because the reactions and article sentiments are contained in the same dataframe

# comparing df2.reactions, df2.article_title_score



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
