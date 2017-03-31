import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats


# Section 1: Focusses on Analysis of Facebook Comments

# Import Comments and Associated Sentiment Calculation
df = pd.read_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-comments-t-sentiment.csv',
                 encoding='utf-8',
                 index_col=0)
# Convert empty na cells to Unicode empty strings -
# Should we not just get ride of them?
# Discuss with Alessandra on how to treat NA data.

# Fill all message columns with no message with blank space
df['message'] = df['message'].fillna(u'')
# Count number of words for each user message
# Calculate the Word Count of each fb user message
df['word_count'] = df['message'].apply(lambda x: len(tokenize.word_tokenize(x)))
# We could remove all NaN elemnts, but instead just iggnore when plotting
# df['sentiment'] = df['sentiment'].fillna(0)


# < 1 > Compare Facebook Comment Length to Sentiment


# Hexagonal Hexbin Plot of Word Count vs FB Comment Sentiment
# Linear Color Bar

x = []
y = []
# Filter out the nan vales and constrain the range
# (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['sentiment'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
hb = ax.hexbin(x, y, gridsize=19)
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, -1, 1])
ax.set_title('Hexagonal Bin Plot of Comment Sentiment vs Comment Word Count'
             '\nfor Facebook Comments in response to Guardian News Articles'
             '\nMax Word Count constrained to 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts', fontsize = 15)
# plt.tight_layout()

# Hexagonal Hexbin Plot
# Logarithmic Color Bar
x = []
y = []
# Filter out the Nan values and constrain the range (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['sentiment'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
hb = ax.hexbin(x, y, gridsize=50, bins='log')
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, -1, 1])
ax.set_title('Hexagonal Bin Plot of Comment Sentiment vs Comment Word Count with Log colour scale'
             '\nfor Facebook Comments in response to Guardian News Articles'
             '\nMax Word Count constrained to 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(Counts)', fontsize = 15)


# < 2 >  Compare Sentiment of Parent sentiment and Word length to parent Comments


# Generate parent arrays for sentiment and word length
# Parents comments can be distinguished by looking at the where the column column_count ! = 0
parentsentiment = df[df['comment_count'] != 0].sentiment.values
parentwordlength = df[df['comment_count'] != 0].word_count.values

# Now, keep only those that have non NaN elemnts for both word length and sentiment
# Return only those terms where they are all real (Remove NaN elements)
# parentindices = parent comments where both parent word length and sentiment are real
parentindices = np.where(~(np.isnan(parentwordlength) | np.isnan(parentsentiment)))[0]
# Extract sentiment and word length, for the above indices
parentsentiment = parentsentiment[parentindices]
parentwordlength = parentwordlength[parentindices]


# Plot histogram, scatter plot, and hexplot for the parent comments
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.hist(parentsentiment, bins = 1000, normed = True)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.hist(parentwordlength, bins =1000, normed = True)
# PLot a hexbin of parent comment word length vs parent comment sentiment
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.hexbin(parentwordlength, parentsentiment, gridsize = 50, bins = 'log')
plt.scatter(parentwordlength, parentsentiment, s = 1)

# < 3 > Compare sentiment of parent comments with sentiment of child comments
# Nan errors still present in data included - consider their inclusion / exclusion with Alessandra

# Examine the correlation between the sentiment of parent comments with
# the associated child comments for that parent comment


# As above, extract Parent comments exist where the comment count != 0
# Generate indices of parent comments
parentindex = np.where(df['comment_count'] != 0)[0]
# Extract sentiment and word length of parent comments, for the above indices
parentsentiment = df[df['comment_count'] != 0].sentiment.values
parentwordlength = df[df['comment_count'] != 0].word_count.values


# Extract all child array comments, separated into index by parent comment
# Here we save all child array comments to an array, where the index
# of the array matches the index of the associated parent comment
# This code says: "Give me all the values between these two indexes"
# where the two indexes are the indexes of the parent comments

# Do this as data is of the form:
# P = Parent, C = Child
# P   C
# 1   0
#     A
#     B
#     C
# 2   0
#     D
#     E
#     F


# Initialise target array for each array of child sentiments
# length of child_sentiments array should equal length of parent_index
# I.e. 1 array for the adult indexes
#      1 array of arrays for the child indexes

parentindex = np.where(df['comment_count'] != 0)[0]
child_sentiments = []
# Iterate over the indexes of parent comments
# and their respective index in the list
for i, index in enumerate(parentindex):
    # Select the last parent - child group first
    # This is done as it needs to be the lat thing to be evaluated.
    # Once it is evaluated, break, to break out of the for loop
    if i == (len(parentindex) - 1):
        print 'last one'
        child_sentiments.append(df['sentiment'][parentindex[-1] + 1:len(df['sentiment']) - 1])
        break
    # Select all other parent - child groups after
    if index < len(df['sentiment']):
        # Select those child comments with index between each parent comment
        child_sentiments.append(df['sentiment'] [parentindex[i]+1:parentindex[i+1]] )

# Plot of all child sentiments for each post (chronological)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
for i in child_sentiments:
    plt.plot(i)
plt.title('Plot of All Child Sentiments, Chronological, and Grouped by Parent Comment')

# Generate mean values for each grouping of child posts
# np.mean() will return NaN if any entry in the list is NaN
# This means that some entries within child_sentiments_mean = NaN
# We must remove these later
child_sentiments_mean = []
for i in child_sentiments:
    child_sentiments_mean.append(np.mean(i))

# Plot the mean values for each post, scatter included
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.plot(child_sentiments_mean)
plt.scatter(np.linspace(0, len(child_sentiments_mean), len(child_sentiments_mean)),
            child_sentiments_mean,
            s=3)
plt.title('Mean Sentiment of Child Comments, grouped by Parents Comment')

# < 4 > Regression Analysis of Parent and Child Comments
#       No removal of outliers done yet
#       Should considder the Removal of child comments
#       where the number of child comments is < than some threshold limit
#       i.e. only include child comments if there are a minimum of ten per parent comment

# Analysis of Correlation between parent comment sentiment and child comment mean group sentiments
# Here we look at the absolute value of sentiment

# IN this way we measure the correlation of the magnitude of the value of sentiment
# not whether it is positive or negative.

# Initialise placeholder arrays for Parent and Child Sentiment
x = []
y = []
# Select only those groups which have no nan values
# See explanation above about np.mean()
for i, v in zip(child_sentiments_mean, parentsentiment):
    if ~np.isnan(i) and ~np.isnan(v):
        x.append(abs(i))
        y.append(abs(v))
# Child Sentiment Mean(Numpy array)
csm = np.array(x)
# Parent Sentiment (Numpy array)
ps = np.array(y)

# Calculate Regression Fit for Parent and Child data
gradient, intercept, r_value, p_value, std_err = stats.linregress(ps, csm)
print('Linear regression using stats.linregress')
fit = ps * gradient + intercept
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.scatter(ps, csm, s = 1, label = 'Scatter Plot Parent vs Child', c = 'b')
plt.plot(ps, fit, label = 'Linear Regression Fit', c='c')
plt.title('Absolute Parent Sentiment vs Absolute Child Sentiment Mean (Grouped) - Linear Regression Fit\n'
          'R 2 = {}'.format(round(r_value ** 2, 4)))
plt.legend()
plt.xlabel('Parent Sentiment (Absolute)')
plt.ylabel('Child Sentiment Mean(Grouped) (Absolute)')
plt.xlim([0, 1])
plt.ylim([0, 1])


# < Section 2 > Focusses on Analysis of Article Sentiment and Message Sentiment

# < 5 > Correlation between article title/message and comments
#       Here we look for correlation between article title sentiment
#                                       and  article message sentiment

# Read the in file containing sentiment on article title and sentiment on article message
df2 = pd.read_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-posts-sentiment.csv', encoding='utf-8', index_col=0)


# Choice between looking at the correlation between the raw -1 - 1 sentiment values
# Here again, we look at the absolute value of sentiment

# Include the following line if you want to remove elemwnts where article title = 0
# df2 = df2[df2.article_title_sentiment != 0]
# df2 = df2[df2.message_sentiment != 0]

# plt.scatter(map(abs, df2['article_title_sentiment']), map(abs, df2['message_sentiment']))

# Use map(abs, XX) to use the absolute values of the sentiment scores
gradient, intercept, r_value, p_value, std_err = stats.linregress(
                                                        map(abs, df2['article_title_sentiment']),
                                                        map(abs, df2['message_sentiment']))
print('Linear regression using stats.linregress')
# Create the linear regression line
fit = np.array(map(abs, df2['article_title_sentiment'])) * gradient + intercept
# Scatter Plot of Absoluute Snetiment Values
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.scatter(map(abs, df2['article_title_sentiment']),
            map(abs, df2['message_sentiment']),
            s = 5,
            label = 'Scatter Plot Title vs Description',
            c = 'b')
# plt.scatter(df2['article_title_sentiment'],
#             df2['message_sentiment'],
#             s = 5,
#             label = 'Scatter Plot Title vs Description',
#             c = 'b')
# Plot the sentiment line of absolute values
plt.plot(map(abs, df2['article_title_sentiment']), fit, label = 'Linear Regression Fit', c='c')
# plt.plot(df2['article_title_sentiment'], fit, label = 'Linear Regression Fit', c='c')
plt.title('Article Title Sentiment vs Article Description Sentiment'
#          '\n Elements where Article Title Sentiment = 0 or Article Description Sentiment = 0 Removed'
          '\nR 2 = {} gradient = {}'.format(round(r_value ** 2, 4), round(gradient ** 2, 4)))
# plt.title('Article Title Sentiment vs Article Description Sentiment')
plt.legend()
plt.xlabel('Article Title Sentiment')
plt.ylabel('Article Description Sentiment')
plt.xlim([0, 1])
plt.ylim([0, 1])


# < 5 b > Comparison of Article Message Sentiment with Facebook Comment Sentiment
# Still have to consider what to do with nans as the np.mean() gonna throw a fucking fit.


# Extract Article Message Sentiment
article_message_sentiment = df2['message_sentiment']
# Extract Article Title Sentiment
article_title_sentiment = df2['article_title_sentiment']
# Extract Article Post Id from Article Info Dataframe
article_post_ids = df2['post_id']

# Post ids in df don't match post ids in df2
# If want to extract Article Post Id's from comment Dataframe, use:
# See issues highlighted in comments
# uniquepostid = pd.Series(df['post_id'].unique()).values

# Used to figure the id out:
# rowstodrop =  np.where(~np.logical_or(df['post_id'] == postid[0], df['post_id'] == postid[1])  )[0]

# Group comment sentiment by post_id

# Initialise target array to store comment sentiment for each post id
# This is comment sentiment, grouped by post_id
article_comments_sentiment = []
# Using the post ids from df2
# Iterate over all article ids extracted from article dataframee
for id in article_post_ids:
    # Return the indexes where the values in post_id in df matches the post_id from df2
    indexes = np.where(df['post_id'] == id)[0]
    # Append all those sentiments that match those indexes
    article_comments_sentiment.append(df['sentiment'].iloc[indexes].values)

# Calculate the mean of the sentiment for extracted comment groups
# Initilise mean sentiment array
article_comments_sentiment_means = []
for sentiments in article_comments_sentiment:
    article_comments_sentiment_means.append( np.mean(sentiments) )

# Non Absolute Plot
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.scatter(article_message_sentiment, article_comments_sentiment_means, label = 'Scatter Plot')
# Absolute Plot (Using the map(abs, XX) functionality discussed:
# plt.scatter(map(abs, article_message_sentiment),
#             map(abs, article_comments_sentiment_means),
#             s = 5,
#             label = 'Article Message Sentiment vs Article Comments Sentiment Means Absolute Values',
#             c = 'b')
plt.legend()
plt.xlabel('Article Message Sentiment ( Not Absolute)')
plt.ylabel('Article Comments Sentiment Means (Not Absolute)')
plt.title('Scatter Plot of Article Message Sentiment vs the Mean of Article Comment Sentiments'
          'Comment Sentiments have been grouped according to the post to which they refer')
plt.xlim([-1, 1])
plt.ylim([-1, 1])

# < Section 3 > Comparison of Facebook Comment sentiment with Article Message Sentiment (Comments not grouped)
#               Article Sentiment (Sentiment of Message and Title)
# Here we will plot article message sentiment against commetn sentiment (with no groups)
# THis should result in  ascatter plot where the number of points is equal to the number of comments
#   (minus the number of comments that were empty and therefore the sentiment = NaN)

# This was implemented to plot sentiment of article message vs all comments associated with that article
# not just the mean sentiment

# Extract the ids for article sentiment and extract the message sentiment where they are.
# ToDo

# A = []
# for i in article_comments_sentiment:
#     A.append(np.zeros(len(i)))
#
#
# # Will need to investigate it like this.
# for i, aa in enumerate(A):
#     print i, aa
#     for j, k in enumerate(aa):
#         print j, k
#         A[i][j] = article_message_sentiment[i]

# new_array = []
# for i,j in enumerate(article_comments_sentiment):
#     new_array.append( [article_message_sentiment[i] for k in ]
# Square 2d Histogram Plot
# Linear
# x = []
# y = []
# for i, v in zip(df['word count'].values, df['sentiment'].values):
#     if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
#         x.append(i)
#         y.append(v)
# # plt.hist2d(x,y, bins=15)
#
# x = np.array(x)
# y = np.array(y)
# fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
# hb = ax.hist2d(x,y, bins = 15)
# ax.axis([0, 100, -1, 1])
# ax.set_title('Hexagonal Bin Plot of Comment Sentiment vs Comment Word Count'
#              '\nfor Facebook Comments in response to Guardian News Articles'
#              '\nMax Word Count constrained to 100', fontsize = 18)
# ax.set_xlabel('Comment Word Count', fontsize = 12)
# ax.set_ylabel('Comment Sentiment (Normalised by Sentence Count)', fontsize = 12)
# # cb = fig.colorbar(hb, ax=ax)
# # cb.set_label('Counts', fontsize = 15)
# plt.tight_layout()

# plt.hist2d(df['word count'], df['sentiment'], bins = 100)
# plt.xlim([0, 100])
# plt.show()
#
# plt.scatter(df['word count'], df['sentiment'], s=1)
# plt.hexbin(df['word count'], df['sentiment'], gridsize = 15)
# plt.xlim([0, 100])
# plt.show()
#
# plt.clf()
#
# x = []
# y = []
# for i,v in zip(df['word count'].values,df['sentiment'].values):
#     if i < 100:
#         x.append(i)
#         y.append(v)
# plt.hexbin(x, y)
# plt.xlim([0, 100])
# plt.clf()
