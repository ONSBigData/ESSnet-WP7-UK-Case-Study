import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import json
import time

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
        # return pd.Series([float('NaN'), float('NaN'), float('NaN'), float('NaN')],
        #                                           index=['Score', 'Pos', 'Neg', 'Neu'])
        return float('NaN'), float('NaN'), float('NaN'), float('NaN')

# < 1 > Import and Add the Large Guardian Facebook Comment file:
# < Import Comment Dataset Comment File >

# Desire to drop dupliates if both the user_id is the same as the message
# Also desire the post_id to be the same (and maybe parent id to be the same)
# Important to make sure you aren't deleting legitimate comments on this.

df = pd.read_csv('C:\Users\cmorris\Documents\wp7 non github\data\comments.csv', sep = ',', encoding = 'utf-8')

df['message_score'] = 0
df['message_pos'] = 0
df['message_neg'] = 0
df['message_neu'] = 0

# Drop nans from the message column
df.dropna(subset=['message'], inplace=True)
# Do sentiment analysis
start_time = time.clock()
df['message_score'], df['message_pos'], df['message_neg'], df['message_neu'] = \
                                                                        zip(*df['message'].map(paragraph_sentiment))
finish_time = time.clock()
print round(finish_time - start_time, 3)
# Find word count of columns
df['word_count'] = df['message'].apply(lambda x: len(tokenize.word_tokenize(x)))
# Output to csv
df.to_csv('C:\Users\cmorris\Documents\wp7 non github\data\comments_test.csv', encoding='utf-8')


# < 2 > Import the post dataset and calculate sentiment

df2 = pd.read_csv('C:\Users\cmorris\Documents\wp7 non github\data\posts.csv', sep=',', encoding='utf-8')
# Drop nans from the message column
df2.dropna(subset=['article_title', 'message'], inplace=True)

start_time = time.clock()
df2['article_message_score'] = 0
df2['article_message_pos'] = 0
df2['article_message_neg'] = 0
df2['article_message_neu'] = 0
df2['article_message_score'], df2['article_message_pos'], df2['article_message_neg'], df2['article_message_neu'] = \
                                                                            zip(*df2.message.map(paragraph_sentiment))
# Calculate Sentiment on the Article Title
# df['article_title_sentiment'] = df.article_title[0:10].apply(lambda x: paragraph_sentiment(x))
df2['article_title_score'] = 0
df2['article_title_pos'] = 0
df2['article_title_neg'] = 0
df2['article_title_neu'] = 0
df2['article_title_score'], df2['article_title_pos'], df2['article_title_neg'], df2['article_title_neu'] = \
                                                                        zip(*df2.article_title.map(paragraph_sentiment))
finish_time = time.clock()
print round(finish_time - start_time, 3)
# Save to new Csv File
df2.to_csv('C:\Users\cmorris\Documents\wp7 non github\data\posts_test.csv', encoding='utf-8')
