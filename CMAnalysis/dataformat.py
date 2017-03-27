import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import json


# < 1 > Import and Add the Guardian Facebook Comment file:

# Read in Line by Line using method posted here:
# http://stackoverflow.com/questions/
# 30088006/loading-a-file-with-more-than-one-line-of-json-into-pythons-pandas
# Loading in using format suggested in:
# http://stackoverflow.com/questions/12451431/loading-and-parsing-a-json-file-with-multiple-json-objects-in-python


data = []
with open('C:/Users/cmorris/PycharmProjects/wp7/data/fb-comments-t.json') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)
# columns = list(df)


# < 2 > Calculate Sentiment for a particular paragraph
def paragraph_sentiment(paragraph):

    # print type(paragraph)
    # print paragraph

    sid = SentimentIntensityAnalyzer()
    # Returns Nan by defiinition in here.
    # That's one of the reasons why it is messing everything up
    # Only calculate sentiment if paragraph is not None:
    if paragraph:
        # print paragraph
        sentences = tokenize.sent_tokenize(paragraph)
        # print sentences
        # print sentences
        score, pos, neg, neu = 0.0, 0.0, 0.0, 0.0
        for sentence in sentences:
            ss = sid.polarity_scores(sentence)
            score = score + ss['compound']
            pos = pos + ss['pos']
            neg = neg + ss['neg']
            neu = neu + ss['neu']

        # print len(sentences)
        numofsents = float(len(sentences))
        normalisedcompoundscore = score / numofsents

        # Normalise each value for the comment paragraph
        ns = score / numofsents
        np = pos / numofsents
        ng = neg / numofsents
        nn = neu / numofsents


        # return normalisedcompoundscore
        # return pd.Series([ns, np, ng, nn], index=['Score', 'Pos', 'Neg', 'Neu'])
        return ns, np, ng, nn

    else:
        # Else, Return Nan if no Paragraph present
        # return float('NaN')
        # return pd.Series([float('NaN'), float('NaN'), float('NaN'), float('NaN')], index=['Score', 'Pos', 'Neg', 'Neu'])
        return float('NaN'), float('NaN'), float('NaN'), float('NaN')
# def f(x):
#     .....:   return pd.Series([x, x ** 2], index=['x', 'x^2'])

# Calculate Sentiment for each Facebook Comment, Return in new column
# df['sentiment'] = df['message'][0:10].apply(lambda x: paragraph_sentiment(x))

df['message_score'] = 0
df['message_pos'] = 0
df['message_neg'] = 0
df['message_neu'] = 0
df['message_score'], df['message_pos'], df['message_neg'], df['message_neu'] = zip(*df['message'].map(paragraph_sentiment))


# Calculate the Word count for each Facebook Comment, Return in new column
df['word count'] = df['message'].apply(lambda x: len(tokenize.word_tokenize(x)))

# Save this new dataframe to csv
df.to_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-comments-t-sentiment-test.csv', encoding='utf-8')
print 'Finish'


# < 3 > Import and Add the Guardian Article Information file:

# Import and save to a dataframe
data = []
with open('C:/Users/cmorris/PycharmProjects/wp7/data/fb-posts.json') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)
columns = list(df)

# Pre Processing needed on the File
# Require sentiment for both article title, and article message
# Don't have article title for every single one,
# so need to ignore those ones in the calculation.

# Pre Proecssing
# Drop all rows with NA in one of the rows
# Discuss this with Alessandra - need to implement a standard for all work.
df.dropna(subset=['article_title', 'message'], inplace=True)


# Calculate Snetiment on the Article Message
# df['message_sentiment'] = df.message[0:10].apply(lambda x: paragraph_sentiment(x))

df['article_message_score'] = 0
df['article_message_pos'] = 0
df['article_message_neg'] = 0
df['article_message_neu'] = 0
df['article_message_score'], df['article_message_pos'], df['article_message_neg'], df['article_message_neu'] = zip(*df.message.map(paragraph_sentiment))


# Calculate Sentiment on the Article Title
# df['article_title_sentiment'] = df.article_title[0:10].apply(lambda x: paragraph_sentiment(x))

df['article_title_score'] = 0
df['article_title_pos'] = 0
df['article_title_neg'] = 0
df['article_title_neu'] = 0
df['article_title_score'], df['article_title_pos'], df['article_title_neg'], df['article_title_neu'] = zip(*df.article_title.map(paragraph_sentiment))


# Save to new Csv File
df.to_csv('C:/Users/cmorris/PycharmProjects/wp7/data/fb-posts-sentiment-test.csv', encoding='utf-8')


# Importing that Didn't Work:
# with open('C:/Users/cmorris/PycharmProjects/wp7/data/fb-comments-t.json', 'rb') as f:
#     data = f.readlines()
# data_json_swtr = "[" + ','.join(data) + "]"
# data_df = pd.read_json(data_json_str)
# Read in The JSON comment file into Pandas
# data = pd.read_json('C:/Users/cmorris/PycharmProjects/wp7/data/fb-comments-t.json')
#                    encoding = 'utf-8')
# Can't use lines = True as this results in a unicode error - can't decode the
# '' in the unicode as per issues highlighted on github here:
# https://github.com/pandas-dev/pandas/issues/15132
