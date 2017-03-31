import pandas as pd
from nltk import tokenize
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import stats
import itertools

# Analysis_posneg focusses on building from the previous work carried out in analysis
# Here we look at analysing not just the overall score but
# the pos, neg, and neu values that went into that score
# This is in an attempt to get more information out of the text
# We are using the new test data files which contain the score, pos, neg, neu for each

# Section 1: Focusses on Analysis of Facebook Comments
# Import Comments and Associated Sentiment Calculation
df = pd.read_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-comments-t-sentiment-test.csv', encoding='utf-8', index_col=0)
# Convert empty na cells to Unicode empty strings -
# Should we not just get ride of them?
# Discuss with Alessandra on how to treat NA data.

# Fill all message columns with no message with blank space
df['message'] = df['message'].fillna(u'')

# Count number of words for each user message
# Already done using pre processing of earlier data
# Calculate the Word Count of each fb user message
df['word_count'] = df['message'].apply(lambda x: len(tokenize.word_tokenize(x)))
# We could remove all NaN elemnts, but instead just iggnore when plotting
# df['sentiment'] = df['sentiment'].fillna(0)

# < Pre - Processing >

# Drop all rows from the parent comments and child comments that have a sentiment of 0 -
df = df[(df.message_score != 0) & (df.message_pos != 0) & (df.message_neg != 0) & (df.message_neu != 1)]
# Drop all rows where there are any NaN values in any of the sentiment scores
df = df[
    (~np.isnan(df.message_score)) |
    (~np.isnan(df.message_pos)) |
    (~np.isnan(df.message_neg)) |
    (~np.isnan(df.message_neu))]

# df = df[df.message_pos != 0]
# df = df[df.message_neg != 0]
# df = df[df.message_neu != 1]

# DO not drop from the article array - as this

# < 1 > Compare Facebook Comment Length to Sentiment
# Hexagonal Hexbin Plot of Word Count vs FB Comment Sentiment
# Linear Color Bar
# Comment Score

x = []
y = []
# Filter out the nan vales and constrain the range
# (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['message_score'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
x_min = 0
x_max = 100
y_min = -1
y_max = 1
hb = ax.hexbin(x, y, gridsize=19, extent=[x_min, x_max, y_min, y_max])
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, -1, 1])
ax.set_title('Hexagonal Bin Plot'
             '\nComment Sentiment (Overall Score) vs Comment Word Count'
             '\nfor parent + child Facebook Comments'
             '\nMax Word Count <= 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Overall Score) Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts', fontsize = 15)
# plt.tight_layout()

# Hexagonal Hexbin Plot
# Logarithmic Color Bar
# Comment Score
x = []
y = []
# Filter out the Nan values and constrain the range (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['message_score'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
x_min = 0
x_max = 100
y_min = -1
y_max = 1
hb = ax.hexbin(x, y, gridsize=50, bins='log', extent=[x_min, x_max, y_min, y_max])
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, -1, 1])
ax.set_title('Hexagonal Bin Plot'
             '\nComment Sentiment (Overall Score) vs Comment Word Count'
             '\nfor parent + child Facebook Comments'
             '\nMax Word Count <= 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Overall Score) Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(Counts)', fontsize = 15)

# Linear Color Bar
# Positive Score

x = []
y = []
# Filter out the nan vales and constrain the range
# (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['message_pos'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
x_min = 0
x_max = 100
y_min = 0
y_max = 1
hb = ax.hexbin(x, y, gridsize=19, extent=[x_min, x_max, y_min, y_max])
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, 0, 1])
ax.set_title('Hexagonal Bin Plot'
             '\nComment Sentiment (Positive Score) vs Comment Word Count'
             '\nfor parent + child Facebook Comments'
             '\nMax Word Count <= 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Positive Score) Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts', fontsize = 15)
# plt.tight_layout()

# Hexagonal Hexbin Plot
# Logarithmic Color Bar
# Message Pos
x = []
y = []
# Filter out the Nan values and constrain the range (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['message_pos'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
x_min = 0
x_max = 100
y_min = 0
y_max = 1
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
hb = ax.hexbin(x, y, gridsize=50, bins='log', extent=[x_min, x_max, y_min, y_max])
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, 0, 1])
ax.set_title('Hexagonal Bin Plot'
             '\nComment Sentiment (Positive Score) vs Comment Word Count'
             '\nfor parent + child Facebook Comments'
             '\nMax Word Count <= 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Positive Score) Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(Counts)', fontsize = 15)

# Linear Color Bar
# Message Neg

x = []
y = []
# Filter out the nan vales and constrain the range
# (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['message_neg'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
x_min = 0
x_max = 100
y_min = 0
y_max = 1
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
hb = ax.hexbin(x, y, gridsize=19, extent=[x_min, x_max, y_min, y_max])
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, 0, 1])
ax.set_title('Hexagonal Bin Plot'
             '\nComment Sentiment (Negative Score) vs Comment Word Count'
             '\nfor parent + child Facebook Comments'
             '\nMax Word Count <= 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Negative Score) Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts', fontsize = 15)
# plt.tight_layout()

# Logarithmic Color Bar
# Message Neg
x = []
y = []
# Filter out the Nan values and constrain the range (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['message_neg'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
x_min = 0
x_max = 100
y_min = 0
y_max = 1
hb = ax.hexbin(x, y, gridsize=50, bins='log', extent=[x_min, x_max, y_min, y_max])
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, 0, 1])
ax.set_title('Hexagonal Bin Plot'
             '\nComment Sentiment (Negative Score) vs Comment Word Count'
             '\nfor parent + child Facebook Comments'
             '\nMax Word Count <= 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Negative Score) Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(Counts)', fontsize = 15)

# Linear Color Bar
# Message Neu

x = []
y = []
# Filter out the nan vales and constrain the range
# (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['message_neu'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
x_min = 0
x_max = 100
y_min = 0
y_max = 1
hb = ax.hexbin(x, y, gridsize=19, extent=[x_min, x_max, y_min, y_max])
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, 0, 1])
ax.set_title('Hexagonal Bin Plot'
             '\nComment Sentiment (Neutral Score) vs Comment Word Count'
             '\nfor parent + child Facebook Comments'
             '\nMax Word Count <= 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Neutral Score) Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Counts', fontsize = 15)
# plt.tight_layout()

# Logarithmic Color Bar
# Message Neu

x = []
y = []
# Filter out the Nan values and constrain the range (hexbin does not do this automatically)
for i, v in zip(df['word_count'].values, df['message_neu'].values):
    if i <= 100 and i > 0 and ~math.isnan(i) and ~math.isnan(v):
        x.append(i)
        y.append(v)
x = np.array(x)
y = np.array(y)
x_min = 0
x_max = 100
y_min = 0
y_max = 1
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
hb = ax.hexbin(x, y, gridsize=50, bins='log', extent=[x_min, x_max, y_min, y_max])
line = ax.plot([0, 0], [0, 0], c='w')
ax.axis([0, 100, 0, 1])
ax.set_title('Hexagonal Bin Plot'
             '\nComment Sentiment (Neutral Score) vs Comment Word Count'
             '\nfor parent + child Facebook Comments'
             '\nMax Word Count <= 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Neutral Score) Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(Counts)', fontsize = 15)

# < 2a > Ddistribution of child comments per parent comment
# Use this to know at what threshold to remove the child / parent comments for analysis
# I.e. if they are outisde of the main 75% threshold then should ignore.

fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
# A = df[df['comment_count'] !=0 ].comment_count.values
A = df[df['comment_count'] > 9].comment_count.values
# A = A[A != 1]
me = np.mean(A)
med = np.median(A)

ax.hist(A,bins = 139, normed = False)
# ax.hist(A,bins = 148, normed = False)
plt.title('Frequency Plot of Number of Child Comments made in response to a single Parent Comment'
          '\nIgnores Parent Comments with less than 10 Child Comments'
          '\nTotal Number of Parent Comments = {}'
          '\nTotal Number of Child Comments = {}'
          '\nMean = {} - Median = {}'.format(len(A),sum(A), np.round(me,2), med))
ax.set_xlabel('Number of Child Comments in response to a single Parent Comment', fontsize = 12)
ax.set_ylabel('Frequency)', fontsize = 12)
# < 2 >  Compare Sentiment of Parent sentiment and Word length to parent Comments


# Generate parent arrays for sentiment and word length
# Parents comments can be distinguished by looking at the where the column column_count ! = 0
parent_message_score = df[df['comment_count'] != 0].message_score.values
parent_message_pos = df[df['comment_count'] != 0].message_pos.values
parent_message_neg = df[df['comment_count'] != 0].message_neg.values
parent_message_neu = df[df['comment_count'] != 0].message_neu.values
parentwordlength = df[df['comment_count'] != 0].word_count.values

# Now, keep only those that have non NaN elemnts for both word length and sentiment
# Return only those terms where they are all real (Remove NaN elements)
# parentindices = parent comments where both parent word length and sentiment are real
parentindices = np.where(~(np.isnan(parentwordlength) | np.isnan(parent_message_score)))[0]
# Extract sentiment and word length, for the above indices
parent_message_score = parent_message_score[parentindices]
parent_message_pos = parent_message_pos[parentindices]
parent_message_neg = parent_message_neg[parentindices]
parent_message_neu = parent_message_neu[parentindices]
parentwordlength = parentwordlength[parentindices]


# Plot histogram, scatter plot, and hexplot for the parent comments
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))
# ax1.add_subplot(1,1,1)
ax1.hist(parent_message_score, bins = 500, normed = False)
ax1.set_xlabel('Sentiment')
ax1.set_ylabel('Frequency')
ax1.set_title('Parent Message Score')

ax2.hist(parent_message_pos, bins = 500, normed = False)
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Frequency')
ax2.set_title('Parent Pos Score')

ax3.hist(parent_message_neg, bins = 500, normed = False)
ax3.set_xlabel('Sentiment')
ax3.set_ylabel('Frequency')
ax3.set_title('Parent Neg Score')

ax4.hist(parent_message_neu, bins = 500, normed = False)
ax4.set_xlabel('Sentiment')
ax4.set_ylabel('Frequency')
ax4.set_title('Parent Neu Score')
plt.suptitle('Frequency Plots for Parent Comment Sentiment Scores')


# PLot a hexbin of parent comment word length vs parent comment sentiment
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
x_min = 0
x_max = 1000
y_min = -1
y_max = 1
plt.hexbin(parentwordlength, parent_message_score, gridsize = 50, bins = 'log', extent=[x_min, x_max, y_min, y_max])
plt.scatter(parentwordlength, parent_message_score, s = 1)

# < 3 > Compare sentiment of parent comments with sentiment of child comments
# Nan errors still present in data included - consider their inclusion / exclusion with Alessandra

# Examine the correlation between the sentiment of parent comments with
# the associated child comments for that parent comment


# As above, extract Parent comments exist where the comment count != 0
# Generate indices of parent comments
parentindex = np.where(df['comment_count'] != 0)[0]
# Extract sentiment and word length of parent comments, for the above indices
parent_message_score = df[df['comment_count'] != 0].message_score.values
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
child_sentiments_pos = []
child_sentiments_neg = []
child_sentiments_neu = []
# Iterate over the indexes of parent comments
# and their respective index in the list
for i, index in enumerate(parentindex):
    # Select the last parent - child group first
    # This is done as it needs to be the lat thing to be evaluated.
    # Once it is evaluated, break, to break out of the for loop
    if i == (len(parentindex) - 1):
        print 'last one'
        child_sentiments.append(df['message_score'][parentindex[-1] + 1:len(df['message_score']) - 1])
        child_sentiments_pos.append(df['message_pos'][parentindex[-1] + 1:len(df['message_score']) - 1])
        child_sentiments_neg.append(df['message_neg'][parentindex[-1] + 1:len(df['message_score']) - 1])
        child_sentiments_neu.append(df['message_neu'][parentindex[-1] + 1:len(df['message_score']) - 1])

        break
    # Select all other parent - child groups after
    if index < len(df['message_score']):
        # Select those child comments with index between each parent comment
        child_sentiments.append(df['message_score'][parentindex[i] + 1:parentindex[i + 1]])
        child_sentiments_pos.append(df['message_pos'][parentindex[i] + 1:parentindex[i + 1]])
        child_sentiments_neg.append(df['message_neg'][parentindex[i] + 1:parentindex[i + 1]])
        child_sentiments_neu.append(df['message_neu'][parentindex[i] + 1:parentindex[i + 1]])

# Plot of all child sentiments for each post (chronological)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
for i in child_sentiments:
    plt.plot(i)
plt.title('Plot of All Child Sentiments, Chronological, and Grouped by Parent Comment')

for i in child_sentiments_pos:
    plt.plot(i)
plt.title('Plot of All Child Sentiments Positive, Chronological, and Grouped by Parent Comment')

for i in child_sentiments_neg:
    plt.plot(i)
plt.title('Plot of All Child Sentiments Negative, Chronological, and Grouped by Parent Comment')

for i in child_sentiments_neu:
    plt.plot(i)
plt.title('Plot of All Child Sentiments Neutral, Chronological, and Grouped by Parent Comment')


# Generate mean values for each grouping of child posts
# np.mean() will return NaN if any entry in the list is NaN
# This means that some entries within child_sentiments_mean = NaN
# We must remove these later
child_sentiments_mean = []
child_sentiments_pos_mean = []
child_sentiments_neg_mean = []
child_sentiments_neu_mean = []

for i, j, k, v in zip(child_sentiments, child_sentiments_pos, child_sentiments_neg, child_sentiments_neu):
    child_sentiments_mean.append(np.mean(i))
    child_sentiments_pos_mean.append(np.mean(j))
    child_sentiments_neg_mean.append(np.mean(k))
    child_sentiments_neu_mean.append(np.mean(v))

# Plot the mean values for each post, scatter included
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.plot(child_sentiments_mean)
plt.plot(child_sentiments_pos_mean)
plt.plot(child_sentiments_neg_mean)
plt.plot(child_sentiments_neu_mean)
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
for i, v in zip(child_sentiments_neu_mean, parent_message_score):
    if ~np.isnan(i) and ~np.isnan(v):
        x.append(i)
        y.append(v)
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
df2 = pd.read_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-posts-sentiment-test.csv', encoding='utf-8', index_col=0)

## Use the test file as it has the pos neg and nue
## and then look at the correlation for pos neg and neu for name nad other

# Choice between looking at the correlation between the raw -1 - 1 sentiment values
# Here again, we look at the absolute value of sentiment

# Include the following line if you want to remove elemwnts where article title = 0
# df2 = df2[df2.article_title_sentiment != 0]
# df2 = df2[df2.message_sentiment != 0]

# plt.scatter(map(abs, df2['article_title_sentiment']), map(abs, df2['message_sentiment']))


# def regression_plot():
#
#
#     return



# Use map(abs, XX) to use the absolute values of the sentiment scores
# gradient, intercept, r_value, p_value, std_err = stats.linregress(
 #                                                       map(abs, df2['article_title_score']),
  #                                                      map(abs, df2['article_message_score']))
gradient, intercept, r_value, p_value, std_err = stats.linregress(
                                                        df2['article_title_pos'],
                                                        df2['article_message_pos'])
print('Linear regression using stats.linregress')
# Create the linear regression line
fit = np.array(df2['article_title_pos']) * gradient + intercept
# Scatter Plot of Absoluute Snetiment Values
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.scatter(df2['article_title_pos'],
            df2['article_message_pos'],
            s = 5,
            label = 'Scatter Plot Title vs Description',
            c = 'b')
# plt.scatter(df2['article_title_sentiment'],
#             df2['message_sentiment'],
#             s = 5,
#             label = 'Scatter Plot Title vs Description',
#             c = 'b')
# Plot the sentiment line of absolute values
plt.plot(df2['article_title_pos'], fit, label = 'Linear Regression Fit', c='c')
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
article_message_sentiment = df2['article_message_score']
# Extract Article Title Sentiment
article_title_sentiment = df2['article_title_score']
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
    article_comments_sentiment.append(df['message_score'].iloc[indexes].values)

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
# ToDo - Add in more here to look at the correlation between article message and facebook message
# A = []
# for i in article_comments_sentiment:
#     A.append(np.zeros(len(i)))
# Will need to investigate it like this.
# for i, aa in enumerate(A):
#     print i, aa
#     for j, k in enumerate(aa):
#         print j, k
#         A[i][j] = article_message_sentiment[i]

##################################
# < 5c > Hexbin plot and linear regression of the positive and negative sentiments
# from the comments sentiment vs article title sentiment



def extractsentiment(dfa, dfc, sat, sct):

    # dfa = dataframe article -> df2
    # dfc = dataframe facebook comment -> df

    # sat = sentiment article type -> options:
    #                               / 'article_message_score'
    #                               / 'article_message_pos'
    #                               / 'article_message_neg'
    #                               / 'article_message_neu'
    # sct = sentiment comment type -> options:
    #                               / 'message_score'
    #                               / 'message_pos'
    #                               / 'message_neg'
    #                               / 'message_neu'

    article_message_sentiment = df2['article_message_score']
    # Extract Article Title Sentiment
    article_title_sentiment = df2['article_title_score']
    # Extract Article Post Id from Article Info Dataframe
    article_post_ids = df2['post_id']

    article_comments_sentiment = []
    # Using the post ids from df2
    # Iterate over all article ids extracted from article dataframee
    for id in article_post_ids:
        # Return the indexes where the values in post_id in df matches the post_id from df2
        indexes = np.where(df['post_id'] == id)[0]
        # Append all those sentiments that match those indexes
        article_comments_sentiment.append(df['message_score'].iloc[indexes].values)
    ###################################
    A = []
    for i in article_comments_sentiment:
        A.append(np.zeros(len(i)))
    for i, aa in enumerate(article_comments_sentiment):
        print 'i: ', i, 'aa: ', aa
        for j, k in enumerate(aa):
            print 'j:', j, 'k: ', k
            A[i][j] = article_message_sentiment.values[i]

    A_flat = list(itertools.chain(*A))
    B_flat = list(itertools.chain(*article_comments_sentiment))

    # ars = article sentiment
    # asf = article sentiment flattened
    # cs = comment sentiment
    # csf = comment sentiment flattened


    return ars, asf, cs, csf

# titlescore, titlescoreflat, articlescore, articlescoreflat = extractsentiment(df2, df, '')


article_message_sentiment = df2['article_message_score']
article_message_pos = df2['article_message_pos']
article_message_neg = df2['article_message_neg']
article_message_neu = df2['article_message_neu']

# Extract Article Title Sentiment
article_title_sentiment = df2['article_title_score']
article_title_pos = df2['article_title_pos']
article_title_neg = df2['article_title_neg']
article_title_neu = df2['article_title_neu']

# Extract Article Post Id from Article Info Dataframe
article_post_ids = df2['post_id']

# Article Comment Arrays
article_comments_sentiment = []
article_comments_pos = []
article_comments_neg = []
article_comments_neu = []

# Using the post ids from df2
# Iterate over all article ids extracted from article dataframee
for id in article_post_ids:
    # Return the indexes where the values in post_id in df matches the post_id from df2
    indexes = np.where(df['post_id'] == id)[0]
    # Append all those sentiments that match those indexes
    article_comments_sentiment.append(df['message_score'].iloc[indexes].values)
    article_comments_pos.append(df['message_pos'].iloc[indexes].values)
    article_comments_neg.append(df['message_neg'].iloc[indexes].values)
    article_comments_neu.append(df['message_neu'].iloc[indexes].values)


# Article Message Arrays
A = []
score = []
pos = []
neg = []
neu = []

# Article Title Arrays
title_score = []
title_pos = []
title_neg = []
title_neu = []


for i in article_comments_sentiment:
    A.append(np.zeros(len(i)))
    score.append(np.zeros(len(i)))
    pos.append(np.zeros(len(i)))
    neg.append(np.zeros(len(i)))
    neu.append(np.zeros(len(i)))

    title_score.append(np.zeros(len(i)))
    title_pos.append(np.zeros(len(i)))
    title_neg.append(np.zeros(len(i)))
    title_neu.append(np.zeros(len(i)))

for i, aa in enumerate(article_comments_sentiment):
    print 'i: ', i, 'aa: ', aa
    for j, k in enumerate(aa):
        print 'j:', j, 'k: ', k
        A[i][j] = article_message_sentiment.values[i]

        # Article Message Sentiment
        score[i][j] = article_message_sentiment.values[i]
        pos[i][j] = article_message_pos.values[i]
        neg[i][j] = article_message_neg.values[i]
        neu[i][j] = article_message_neu.values[i]

        # Article Title Sentiment
        title_score[i][j] = article_title_sentiment.values[i]
        title_pos[i][j] = article_title_pos.values[i]
        title_neg[i][j] = article_title_neg.values[i]
        title_neu[i][j] = article_title_neu.values[i]

# Create flat arrays for hexbin plot

# Article Message flat
# score_flat = list(itertools.chain(*score))
# pos_flat = list(itertools.chain(*pos))
# neg_flat = list(itertools.chain(*neu))
#neu_flat = list(itertools.chain(*neg))

# Article Title flat
score_flat = list(itertools.chain(*title_score))
pos_flat = list(itertools.chain(*title_pos))
neg_flat = list(itertools.chain(*title_neu))
neu_flat = list(itertools.chain(*title_neg))

# Comment flat
B_flat = list(itertools.chain(*article_comments_sentiment))
article_comments_sentiment_flat = list(itertools.chain(*article_comments_sentiment))
article_comments_pos_flat = list(itertools.chain(*article_comments_pos))
article_comments_neg_flat = list(itertools.chain(*article_comments_neg))
article_comments_neu_flat = list(itertools.chain(*article_comments_neu))

# Subplot of all histograms / scatter / hexbin plots Same layout as before using the subplot notation

# plt.scatter(A_flat, B_flat)
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.hexbin(score_flat, article_comments_sentiment_flat, gridsize=100, bins='log')
plt.hexbin(pos_flat, article_comments_pos_flat, gridsize=100, bins='log')
plt.hexbin(neg_flat, article_comments_neg_flat, gridsize=100, bins='log')
plt.hexbin(neu_flat, article_comments_neu_flat, gridsize=100, bins='log')
plt.xlim([0,1])
cb = fig.colorbar(hb, ax=ax)
######################################


# Plotting all the comment sentiments for sentiment for each article title
x_min = -1
x_max = 1
y_min = -1
y_max = 1
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(7, 4))
# ax1.add_subplot(1,1,1)
a1 = ax1.hexbin(score_flat, article_comments_sentiment_flat, gridsize=100, bins='log', extent=[x_min, x_max, y_min, y_max])
ax1.set_xlabel('Article Title Sentiment Score')
ax1.set_ylabel('Facebook Comment Score')
ax1.set_title('Article Title Sentiment vs Facebook Comment Sentiment Score')
ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])
cb1 = fig.colorbar(a1, ax=ax1)
cb1.set_label('log10(Counts)', fontsize = 15)

x_min = -1
x_max = 1
y_min = -1
y_max = 1
a2 = ax2.hexbin(pos_flat, article_comments_pos_flat, gridsize=100, bins='log', extent=[x_min, x_max, y_min, y_max])
ax2.set_xlabel('Article Title Sentiment Score')
ax2.set_ylabel('Facebook Comment Score')
ax2.set_title('Article Title Sentiment vs Facebook Comment Sentiment Positive')
ax2.set_xlim([-1,1])
ax2.set_ylim([-1,1])
cb2 = fig.colorbar(a2, ax=ax2)
cb2.set_label('log10(Counts)', fontsize = 15)

a3 = ax3.hexbin(neg_flat, article_comments_neg_flat, gridsize=100, bins='log', extent=[x_min, x_max, y_min, y_max])
ax3.set_xlabel('Article Title Sentiment Score')
ax3.set_ylabel('Facebook Comment Score')
ax3.set_title('Article Title Sentiment vs Facebook Comment Sentiment Negative')
ax3.set_xlim([-1,1])
ax3.set_ylim([-1,1])
cb3 = fig.colorbar(a3, ax=ax3)
cb3.set_label('log10(Counts)', fontsize = 15)

a4 = ax4.hexbin(neu_flat, article_comments_neu_flat, gridsize=100, bins='log', extent=[x_min, x_max, y_min, y_max])
ax4.set_xlabel('Article Title Sentiment Score')
ax4.set_ylabel('Facebook Comment Score')
ax4.set_title('Article Title Sentiment vs Facebook Comment Sentiment Neutral')
ax4.set_xlim([-1,1])
ax4.set_ylim([-1,1])
cb4 = fig.colorbar(a4, ax=ax4)
cb4.set_label('log10(Counts)', fontsize = 15)
# cb = fig.colorbar(hb, ax=ax)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)
# cb.set_label('log10(Counts)', fontsize = 15)


### Further Histogram to highlight the differences in pos neg and neu which should all be on the same axis
x_min = 0
x_max = 1
y_min = 0
y_max = 1
fig, (ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=3, figsize=(7, 4))
a2 = ax2.hexbin(pos_flat, article_comments_pos_flat, gridsize=100, bins='log', extent=[x_min, x_max, y_min, y_max])
ax2.set_xlabel('Article Title Sentiment Score')
ax2.set_ylabel('Facebook Comment Score')
ax2.set_title('Article Title Sentiment vs Facebook Comment Sentiment Positive')
ax2.set_xlim([0,1])
ax2.set_ylim([0,1])
cb2 = fig.colorbar(a1, ax=ax2)
cb2.set_label('log10(Counts)', fontsize = 15)

a3 = ax3.hexbin(neg_flat, article_comments_neg_flat, gridsize=100, bins='log', extent=[x_min, x_max, y_min, y_max])
ax3.set_xlabel('Article Title Sentiment Score')
ax3.set_ylabel('Facebook Comment Score')
ax3.set_title('Article Title Sentiment vs Facebook Comment Sentiment Negative')
ax3.set_xlim([0,1])
ax3.set_ylim([0,1])
cb3 = fig.colorbar(a1, ax=ax3)
cb3.set_label('log10(Counts)', fontsize = 15)

a4 = ax4.hexbin(neu_flat, article_comments_neu_flat, gridsize=100, bins='log', extent=[x_min, x_max, y_min, y_max])
ax4.set_xlabel('Article Title Sentiment Score')
ax4.set_ylabel('Facebook Comment Score')
ax4.set_title('Article Title Sentiment vs Facebook Comment Sentiment Neutral')
ax4.set_xlim([0,1])
ax4.set_ylim([0,1])
cb4 = fig.colorbar(a1, ax=ax4)
cb4.set_label('log10(Counts)', fontsize = 15)


for i, j in zip(A, article_comments_sentiment):
    plt.plot(i,j)

# So the line plot works ok, but a hexbin plot would allow further information on what the density
# of certain parts of the graph are.
A_flat = list(itertools.chain(*A))
B_flat = list(itertools.chain(*article_comments_sentiment))

fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.scatter(A_flat,B_flat)
plt.hexbin(A_flat,B_flat, gridsize=100, bins='log')
plt.hexbin(A_flat,B_flat, gridsize=100, bins='log')
cb = fig.colorbar(hb, ax=ax)

# Correlation between the article title sentiment and the facebook comment sentiment
# Can track how this changes by the global data changes made at the top of the file

# Linear Regression for Positive Sentiment Scores
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
gradient, intercept, r_value, p_value, std_err = stats.linregress(
                                                        pos_flat,
                                                        article_comments_pos_flat,
                                                        )
print('Linear regression using stats.linregress')
# Create the linear regression line
fit = np.array(pos_flat) * gradient + intercept
# Scatter Plot of Absoluute Snetiment Values
plt.scatter(pos_flat,article_comments_pos_flat,
            s = 5,
            label = 'Scatter Plot Title Sentiment vs Comment Sentiment',
            c = 'b')
# plt.scatter(df2['article_title_sentiment'],
#             df2['message_sentiment'],
#             s = 5,
#             label = 'Scatter Plot Title vs Description',
#             c = 'b')
# Plot the sentiment line of absolute values
plt.plot(pos_flat, fit, label = 'Linear Regression Fit', c='c')
# plt.plot(df2['article_title_sentiment'], fit, label = 'Linear Regression Fit', c='c')
plt.title('Article Title Sentiment vs Comment Sentiment (Positive Sentiment)'
#          '\n Elements where Article Title Sentiment = 0 or Article Description Sentiment = 0 Removed'
          '\nR 2 = {} gradient = {}'.format(round(r_value ** 2, 4), round(gradient ** 2, 4)))
# plt.title('Article Title Sentiment vs Article Description Sentiment')
plt.legend()
plt.xlabel('Article Title Sentiment')
plt.ylabel('Article Comments Sentiment')
plt.xlim([0, 1])
plt.ylim([0, 1])

# Linear Regression for Negative Sentiment Scores
# Create the linear regression line
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
gradient, intercept, r_value, p_value, std_err = stats.linregress(
                                                        neg_flat,
                                                        article_comments_neg_flat,
                                                        )
fit = np.array(neg_flat) * gradient + intercept
# Scatter Plot of Absoluute Snetiment Values
plt.scatter(neg_flat,article_comments_neg_flat,
            s = 5,
            label = 'Scatter Plot Title Sentiment vs Comment Sentiment',
            c = 'b')
# plt.scatter(df2['article_title_sentiment'],
#             df2['message_sentiment'],
#             s = 5,
#             label = 'Scatter Plot Title vs Description',
#             c = 'b')
# Plot the sentiment line of absolute values
plt.plot(neg_flat, fit, label = 'Linear Regression Fit', c='c')
# plt.plot(df2['article_title_sentiment'], fit, label = 'Linear Regression Fit', c='c')
plt.title('Article Title Sentiment vs Comment Sentiment (Negative Sentiment)'
#          '\n Elements where Article Title Sentiment = 0 or Article Description Sentiment = 0 Removed'
          '\nR 2 = {} gradient = {}'.format(round(r_value ** 2, 4), round(gradient ** 2, 4)))
# plt.title('Article Title Sentiment vs Article Description Sentiment')
plt.legend()
plt.xlabel('Article Title Sentiment')
plt.ylabel('Article Comments Sentiment')
plt.xlim([0, 1])
plt.ylim([0, 1])

# Linear Regression for Neutral Sentiment Scores
# Create the linear regression line
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
gradient, intercept, r_value, p_value, std_err = stats.linregress(
                                                        neu_flat,
                                                        article_comments_neu_flat)
fit = np.array(neu_flat) * gradient + intercept
# Scatter Plot of Absoluute Snetiment Values
plt.scatter(neu_flat,article_comments_neu_flat,
            s = 5,
            label = 'Scatter Plot Title Sentiment vs Comment Sentiment',
            c = 'b')
# plt.scatter(df2['article_title_sentiment'],
#             df2['message_sentiment'],
#             s = 5,
#             label = 'Scatter Plot Title vs Description',
#             c = 'b')
# Plot the sentiment line of absolute values
plt.plot(neu_flat, fit, label = 'Linear Regression Fit', c='c')
# plt.plot(df2['article_title_sentiment'], fit, label = 'Linear Regression Fit', c='c')
plt.title('Article Title Sentiment vs Comment Sentiment (Neutral Sentiment)'
#          '\n Elements where Article Title Sentiment = 0 or Article Description Sentiment = 0 Removed'
          '\nR 2 = {} gradient = {}'.format(round(r_value ** 2, 4), round(gradient ** 2, 4)))
# plt.title('Article Title Sentiment vs Article Description Sentiment')
plt.legend()
plt.xlabel('Article Title Sentiment')
plt.ylabel('Article Comments Sentiment')
plt.xlim([0, 1])
plt.ylim([0, 1])
