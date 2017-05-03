import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats


comments['word_count'] = comments['message'].apply(lambda x: len(tokenize.word_tokenize(x)))


# < 1 > Compare Facebook Comment Length to Sentiment

# Hexagonal Hexbin Plot - up to 100 word count
# Logarithmic Color Bar
x = np.array(comments[(comments['word_count'] > 0) & (comments['word_count'] <= 800)]['word_count'])
y = np.array(sentiment[(comments['word_count'] > 0) & (comments['word_count'] <= 800)]['vader'])
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))

xmin, xmax, ymin, ymax = 0, 100, -1, 1

hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='viridis', extent = [xmin, xmax, ymin, ymax])
ax.axis([xmin, xmax, ymin, ymax])
ax.set_title('Hexagonal Bin Plot of Comment Sentiment vs Comment Word Count with Log colour scale'
             '\nfor Facebook Comments in response to Guardian News Articles'
             '\nMax Word Count constrained to 100', fontsize = 18)
ax.set_xlabel('Comment Word Count', fontsize = 12)
ax.set_ylabel('Comment Sentiment (Normalised by Sentence Count)', fontsize = 12)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(Counts)', fontsize = 15)


# < 2 > Correlation between article title/message
#       Here we look for correlation between article title sentiment
#                                       and  article message sentiment


valid_posts = posts[~posts['article_title'].isnull()].copy()
valid_posts['article_title_sentiment'] = valid_posts.article_title.apply(paragraph_sentiment)
valid_posts['message_sentiment'] = valid_posts.message.apply(paragraph_sentiment)
# Use map(abs, XX) to use the absolute values of the sentiment scores
gradient, intercept, r_value, p_value, std_err = stats.linregress(valid_posts['article_title_sentiment'],
                                                                  valid_posts['message_sentiment'])
print('Linear regression using stats.linregress')
# Create the linear regression line
fit = np.array(valid_posts['article_title_sentiment']) * gradient + intercept
# Scatter Plot of Absoluute Snetiment Values
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.scatter(valid_posts['article_title_sentiment'],
            valid_posts['message_sentiment'],
            s = 5,
            label = 'Scatter Plot Title vs Description',
            c = 'b')
plt.plot(valid_posts['article_title_sentiment'], fit, label = 'Linear Regression Fit', c='c')
plt.title('Article Title Sentiment vs Article Description Sentiment'
          '\nR 2 = {} gradient = {}'.format(round(r_value ** 2, 4), round(gradient ** 2, 4)))
plt.legend()
plt.xlabel('Article Title Sentiment')
plt.ylabel('Article Description Sentiment')
plt.xlim([-1, 1])
plt.ylim([-1, 1])



# < 3 > Correlation between article title and facebook comments (grouped)

valid_posts.set_index('post_id', inplace=True)
all_comments = pd.concat([comments, sentiment['vader']], axis = 1)
all_comments = all_comments[['post_id', 'vader']].groupby('post_id').mean()
combined = pd.merge(valid_posts[['article_title', 'article_title_sentiment']], all_comments, left_index=True, right_index=True, how='inner')

x = np.array(combined['article_title_sentiment'])
y = np.array(combined['vader'])

gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print('Linear regression using stats.linregress')
fit = x * gradient + intercept
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.scatter(x, y, s = 1, label = 'Scatter Plot Article Title vs Comments', c = 'b')
plt.plot(x, fit, label = 'Linear Regression Fit', c='c')
plt.title('Article Title Sentiment vs Comments Sentiment (Grouped) - Linear Regression Fit\n'
          'R 2 = {}'.format(round(r_value ** 2, 4)))
plt.legend()
plt.xlabel('Article Title Sentiment')
plt.ylabel('Comments Sentiment Mean(Grouped)')
plt.xlim([-1, 1])
plt.ylim([-1, 1])




# < 4 > Correlation between parent and childs (grouped) comments

childs = pd.concat([comments[~comments['parent_id'].isnull()], sentiment[~comments['parent_id'].isnull()]['vader']], axis = 1)
childs = childs[['parent_id', 'vader']].groupby('parent_id').mean()
parents = pd.concat([comments[comments['parent_id'].isnull()], sentiment[comments['parent_id'].isnull()]['vader']], axis = 1)
parents.set_index('comment_id', inplace = True)
relatives = pd.merge(parents[['post_id', 'vader']], childs, left_index=True, right_index=True, how='inner', suffixes=('_parent', '_child'))

# Calculate Regression Fit for Parent and Child data
x = np.array(relatives['vader_parent'])
y = np.array(relatives['vader_child'])

gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print('Linear regression using stats.linregress')
fit = x * gradient + intercept
fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
plt.scatter(x, y, s = 1, label = 'Scatter Plot Parent vs Child', c = 'b')
plt.plot(x, fit, label = 'Linear Regression Fit', c='c')
plt.title('Parent Sentiment vs Child Sentiment Mean (Grouped) - Linear Regression Fit\n'
          'R 2 = {}'.format(round(r_value ** 2, 4)))
plt.legend()
plt.xlabel('Parent Sentiment')
plt.ylabel('Child Sentiment Mean(Grouped)')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
