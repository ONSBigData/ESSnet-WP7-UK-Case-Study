import pandas as pd
from nltk import tokenize
from matplotlib import pyplot as plt
import numpy as np
import itertools
import ast

# Import Facebook Comments (on Articles) Short Datafile
# df = pd.read_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-comments-t-sentiment-test.csv',
#                  encoding='utf-8',
#                  index_col=0)

# Import Facebook Comments (on Articles) Full Datafile
df = pd.read_csv('C:\Users\cmorris\Documents\wp7 non github\data\comments_final_test.csv',
                 encoding='utf-8',
                 index_col=0)

# Fill nan values with empty space
# Should be no nan values at this stage as should have been removed
df['message'] = df['message'].fillna(u'')
df['word_count'] = df['message'].apply(lambda x: len(tokenize.word_tokenize(x)))

# Import Article Messages / Title Datafile
# df2 = pd.read_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-posts-sentiment-test.csv', encoding='utf-8', index_col=0)

# Import Article Posts Messages Full Datafile
df2 = pd.read_csv('C:\Users\cmorris\Documents\wp7 non github\data\posts_final_test.csv',
                 encoding='utf-8',
                 index_col=0)

# Extract Article Title Sentiments
article_title_score = df2.article_title_score.values
article_title_pos = df2.article_title_pos.values
article_title_neu = df2.article_title_neg.values
article_title_neu = df2.article_title_neu.values


# Err: DOn't need the following lines when dealing with the full datat set as reactions explicitly defined.
# Instead renmae the columns to fit with current titles used in this work
# Extract Article Reactions from the column containing a dictionary of reactions
# from the retarded pandas unicode format
# Store Reactions in its own dataframe

# Merge Reaction DataFrame with Article Message Dataframe
# Drop index as missing some indices cuasing concat not to line everything up appropriate

# df2.reactions = df2.reactions.apply(lambda x: ast.literal_eval(x))
# reactions = df2.reactions.values.tolist()
# rdf = pd.DataFrame(reactions)
# df2.reset_index(drop=False, inplace=True)
# df2 = pd.concat([df2, rdf], ignore_index=False, axis = 1)

# Rename Columns

df2.rename(columns={'reactions.angry': 'angry',
                   'reactions.haha': 'haha',
                   'reactions.like': 'like',
                   'reactions.love': 'love',
                   'reactions.sad': 'sad',
                   'reactions.thankful': 'thankful',
                   'reactions.total_count': 'total_count',
                   'reactions.wow': 'wow'}, inplace=True)


# < 6 > Correlation between the reactions to an article and the sentiment of the article title

# Article Title Sentiment contained between df2.columns[23:27]
# Article Message Sentiment constained between df2.columns[29:23]
# Reactions contained in columns [9:17]

# Correlation between Article Title Sentiment and Reactions
# Basic Scatter Graphs for each sentiment
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))
for column in df2.columns[9:17]:
    ax1.scatter(df2.article_title_score,df2['{}'.format(column)], label = column)
for column in df2.columns[9:17]:
    ax2.scatter(df2.article_title_pos,df2['{}'.format(column)], label = column)
for column in df2.columns[9:17]:
    ax3.scatter(df2.article_title_neg,df2['{}'.format(column)], label = column)
for column in df2.columns[9:17]:
    ax4.scatter(df2.article_title_neu,df2['{}'.format(column)], label = column)
ax1.set_xlabel('Article Sentiment')
ax1.set_ylabel('Article Reaction')
plt.xlabel('Article Sentiment')
plt.ylabel('Article Reaction')
plt.legend()

# Correlation between Article Title Sentiment and Reactions
# Scatter plot of Reactions All on Same Plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))
# Flatten axes to effecively iterate over them
for axis, sentiment in zip(list(itertools.chain(*axes)), df2.columns[23:27]):
    print axis
    print sentiment
    #for ax in axis:
    for column in df2.columns[9:17]:
        axis.scatter(df2['{}'.format(sentiment)],df2['{}'.format(column)], label = column, s = 4)
    axis.set_xlabel('Sentiment')
    axis.set_ylabel('Frequency')
    axis.set_title('Article Title Sentiment vs Reaction Frequency - {}'.format(sentiment))
# Add a common legend outside of the plots
plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
plt.suptitle('Article Title Sentiment vs Reaction Frequency')

# Article Title/Message Sentiment vs Reactions Hexbin Plot on Separate Plots
# Column locations for Title Sentiment
# Article Title Sentiment contained between df2.columns[17:21]
# Article Message Sentiment constained between df2.columns[13:17]
for reaction in df2.columns[9:17]:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))

    #max_reaction = 0
    #if max(df2['{}'.format(reaction)]) > max_reaction:
    #    max_reaction = max(df2['{}'.format(reaction)])
    #print max_reaction



    # Article Message Sentiment:
    # for axis, sentiment in zip(list(itertools.chain(*axes)), df2.columns[13:17]):

    # Article Title Sentiment:
    for axis, sentiment in zip(list(itertools.chain(*axes)), df2.columns[23:27]):

        print axis
        print sentiment

        #for ax in axis:

        # axis.scatter(df2['{}'.format(sentiment)],df2['{}'.format(reaction)], label = reaction, s = 4)
        axis.hexbin(df2['{}'.format(sentiment)],df2['{}'.format(reaction)], gridsize=25, bins='log')
        axis.set_xlabel('{} Sentiment'.format(sentiment))
        axis.set_ylabel('Frequency')
        axis.set_title('{} vs {} Frequency '.format(sentiment, reaction))
        axis.set_xlim([0,1])
        axis.set_ylim([0, 50000])
    list(itertools.chain(*axes))[0].set_xlim([-1,1])
    plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    # Title Sentiment
    # plt.suptitle('Article Title Sentiment vs {} Frequency'.format(reaction))
    # Message Sentiment
    plt.suptitle('Article Message Sentiment vs {} Frequency'.format(reaction))


# < 7 >
# Comparison of Article Reactions and Sentiment of Facebook Comments
# Drop any nan instances in any of the comment message scores
df.dropna(subset = ['message_score', 'message_pos', 'message_neg','message_neu'], inplace = True)
# Drop all rows from the parent comments and child comments that have a sentiment of 0 -
df = df[(df.message_score != 0) & (df.message_pos != 0) & (df.message_neg != 0) & (df.message_neu != 1)]

# Create arrays containing the comments and reactions
# Comment Arrays
indexes = []
article_comments_sentiment = []
article_comments_pos = []
article_comments_neg = []
article_comments_neu = []

for index in df2['post_id']:
    i = np.where(df['post_id'] == index)[0]
    indexes.append(np.where(df['post_id'] == index)[0])
    article_comments_sentiment.append(df['message_score'].iloc[i].values)
    article_comments_pos.append(df['message_pos'].iloc[i].values)
    article_comments_neg.append(df['message_neg'].iloc[i].values)
    article_comments_neu.append(df['message_neu'].iloc[i].values)

# Reaction Arryas
angry = []
haha = []
like = []
love = []
sad = []
thankful = []
total_count = []
wow = []

for i in indexes:
    angry.append(np.zeros(len(i)))
    haha.append(np.zeros(len(i)))
    like.append(np.zeros(len(i)))
    love.append(np.zeros(len(i)))
    sad.append(np.zeros(len(i)))
    thankful.append(np.zeros(len(i)))
    total_count.append(np.zeros(len(i)))
    wow.append(np.zeros(len(i)))

for i, aa in enumerate(indexes):
    print 'i: ', i, 'aa: ', aa
    for j, k in enumerate(aa):
        print 'j:', j, 'k: ', k
        angry[i][j] = df2.angry.values[i]
        haha[i][j] = df2.haha.values[i]
        like[i][j] = df2.like.values[i]
        love[i][j] = df2.love.values[i]
        sad[i][j] = df2.sad.values[i]
        thankful[i][j] = df2.thankful.values[i]
        total_count[i][j] = df2.total_count.values[i]
        wow[i][j] = df2.wow.values[i]


# Now, as everything is all lined up
# flatten the lists and plot the two in a hexplot
angry = list(itertools.chain(*angry))
haha = list(itertools.chain(*haha))
like = list(itertools.chain(*like))
love = list(itertools.chain(*love))
sad = list(itertools.chain(*sad))
thankful = list(itertools.chain(*thankful))
total_count = list(itertools.chain(*total_count))
wow = list(itertools.chain(*wow))

article_comments_sentiment = list(itertools.chain(*article_comments_sentiment))
article_comments_pos = list(itertools.chain(*article_comments_pos))
article_comments_neg = list(itertools.chain(*article_comments_neg))
article_comments_neu = list(itertools.chain(*article_comments_neu))

# Define function for hexplot
def hexplot(x,y, xtitle, ytitle, title):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    # plt.hexbin(total_count,article_comments_pos, gridsize=100, bins='log')
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = max(y)
    a1 = axes.hexbin(x, y, gridsize=75, bins = 'log', extent=[x_min, x_max, y_min, y_max])
    cb2 = fig.colorbar(a1, ax=axes)
    cb2.set_label('log10(Counts)', fontsize=15)
    axes.set_xlabel('{}'.format(xtitle))
    axes.set_ylabel('{}'.format(ytitle))
    axes.set_title('{}'.format(title))
    axes.set_xlim([0, 1])
    axes.set_ylim([0, max(y)])
    plt.show()
    return 1

# Create hexplot for
hexplot(article_comments_pos,
        like,
        'Comment Sentiment (Positive)',
        'Reaction Frequency - Total Count',
        'Hexbin Plot Positive Sentiment vs total_count frequency'
        '\nShading indicates frequency')

# Can remove all instances where all comment sentiments are 0
# i.e. a question what influences when someone posts a comment that does not have 0 sentiment attached to it


# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
# plt.hexbin(total_count,article_comments_pos, gridsize=100, bins='log')
# plt.hexbin(article_comments_pos, total_count, gridsize=13, bins='log')

# Scatter Plots
# for i, v in zip(angry, article_comments_pos):
#     print i,v
#     # plt.scatter(i,v)
#     plt.scatter(v, i)

# Get the article id from df2 for all articles
# Return a list of indices of comments for a particular article Id
# Return a list of comment sentiment for those indices
# Compare the comment sentiments to the sentiments of the articles


# df2.reactions[0:1].values[0]
# reactions = ast.literal_eval(df2.reactions[0:1].values[0])
# type(ast.literal_eval(df2.reactions[0:1].values[0]))
# df2.reactions = df2.reactions.apply(lambda x: ast.literal_eval(x))
# reactions = df2.reactions.values
