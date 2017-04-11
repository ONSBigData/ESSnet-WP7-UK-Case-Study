from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from nltk.sentiment.vader import normalize
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from nltk import sent_tokenize

df = pd.read_csv('fb-comments-t-sentiment.csv', encoding='utf-8')
df = df[df.message.notnull()]
df['n_sents'] = df.message.apply(lambda x: len(sent_tokenize(x)))
df.set_index(pd.to_datetime(df['created_time']), inplace = True)
# Load all others dictionaries
bing = pd.read_csv("bing.csv")
bing.drop_duplicates('word', inplace = True)

afinn = pd.read_csv("afinn.csv")
afinn.drop_duplicates('word', inplace = True)

syuzhet = pd.read_csv("syuzhet.csv")
syuzhet.drop_duplicates('word', inplace = True)

nrc = pd.read_csv("nrc.csv", header=0, names = [u'word', u'anger', u'anticipation', u'disgust', u'fear', u'joy',
       u'negative', u'positive', u'sadness', u'surprise', u'trust'])
nrc.drop_duplicates('word', inplace = True)
nrc['value'] = nrc['positive']-nrc['negative']

def rescale(x): # it doesn't guarantee sign is kept
    return  2 * (x - min(x))/(max(x) - min(x)) - 1



def get_scores(text, vocab):
    vectorizer = CountVectorizer(vocabulary=vocab['word'])
    vect = vectorizer.fit_transform(text)
    values = np.array(vocab['value'])
    scores  = vect.dot(values)
    return scores


def get_nrc_scores(text):
    vectorizer = CountVectorizer(vocabulary=nrc['word'])
    vect = vectorizer.fit_transform(text)
    values = nrc[['positive','negative']].as_matrix()
    scores  = vect.dot(values)
    return scores

def get_nrc_emotions(text, plot_dist=False):
    vectorizer = CountVectorizer(vocabulary=nrc['word'])
    vect = vectorizer.fit_transform(text)
    colnames = [u'anger', u'anticipation', u'disgust',
                u'fear', u'joy', u'sadness', u'surprise',
                u'trust']
    values = csr_matrix(nrc[colnames].values)
    scores = vect.dot(values)
    if plot_dist:
        for emotion in colnames:
            vocabulary = nrc[nrc[emotion] > 0]['word']
            freq = vect[:, vocabulary]
    return pd.DataFrame(scores.todense(), columns=colnames)


bing_scores = get_scores(df['message'], bing)
afinn_scores = get_scores(df['message'], afinn)
syuzhet_scores = get_scores(df['message'], syuzhet)
nrc_scores = get_scores(df['message'], nrc)

all_methods = pd.DataFrame({'bing': bing_scores,
              'afinn': afinn_scores,
              'syuzhet': syuzhet_scores,
              'nrc': nrc_scores},
              index=pd.to_datetime(df['created_time'])).div(df.n_sents, axis='index')

simple_rescaled = all_methods.apply(rescale)

simple_rescaled.plot()

vader_rescaled = all_methods.apply(lambda x: map(normalize, x))

vader_rescaled[vader_rescaled['afinn'] != 0]['afinn'].plot()
vader_rescaled[vader_rescaled['afinn'] != 0]['afinn'].resample('H').mean().plot()

vader_rescaled[vader_rescaled['afinn'] > 0]['afinn'].resample('4H').mean().plot(legend=True, label='Positives')
vader_rescaled[vader_rescaled['afinn'] < 0]['afinn'].resample('4H').mean().plot(legend=True, label='Negatives')

emotions = get_nrc_emotions(df['message'])

emotions.set_index(pd.to_datetime(df['created_time'])).plot()
emotions.set_index(pd.to_datetime(df['created_time'])).resample('H').mean().plot()

df['created_time']=pd.to_datetime(df['created_time'])
df.set_index(pd.to_datetime(df.index), inplace=True)
df.resample('H').count()['comment_id'].plot(legend=True, label='Volume')

# Correlation matrix between methods
simple_rescaled.corr()

simple_rescaled.resample('.5*H').mean().plot()
vader_rescaled.resample('H').mean().plot()

from sklearn.metrics import classification_report
validated = pd.read_csv("validated_gd.csv", index_col=0, parse_dates=True, encoding="utf-8")
bing_scores = get_scores(validated['message'], bing)
afinn_scores = get_scores(validated['message'], afinn)
syuzhet_scores = get_scores(validated['message'], syuzhet)
nrc_scores = get_scores(validated['message'], nrc)

validated['n_sents'] = validated.message.apply(lambda x: len(sent_tokenize(x)))
all_methods = pd.DataFrame({'bing': bing_scores,
              'afinn': afinn_scores,
              'syuzhet': syuzhet_scores,
              'nrc': nrc_scores},
              index=validated.index).div(validated.n_sents, axis='index')

all_methods = all_methods.apply(lambda x: map(normalize, x))


validated = pd.concat([validated, all_methods], axis=1)
validated['vader'] = validated.message.apply(paragraph_sentiment)

validated[['A_Score', 'afinn', 'bing', 'nrc', 'syuzhet', 'vader']].corr()

f = lambda r: ['P' if x > 0.2 else 'N' if x < -0.2 else 'X' for x in r]
sentiment = validated[['A_Score', 'afinn', 'bing', 'nrc', 'syuzhet', 'vader']].apply(f)
sentiment.columns = ['Manual', 'Afinn', 'Bing', 'NRC', 'Syuzhet', 'Vader']
for method in sentiment.columns[1:]:
    print "Print classification metrics for method %s ...... " %method
    print classification_report(sentiment['Manual'], sentiment[method])


from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def get_scores_with_lemma(text, vocab):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), vocabulary=vocab['word'])
    vect = vectorizer.fit_transform(text)
    values = np.array(vocab['value'])
    scores  = vect.dot(values)
    return scores


bing_scores_l = get_scores_with_lemma(validated['message'], bing)
afinn_scores_l = get_scores_with_lemma(validated['message'], afinn)
syuzhet_scores_l = get_scores_with_lemma(validated['message'], syuzhet)
nrc_scores_l = get_scores_with_lemma(validated['message'], nrc)

all_methods = pd.DataFrame({'bing': bing_scores_l,
              'afinn': afinn_scores_l,
              'syuzhet': syuzhet_scores_l,
              'nrc': nrc_scores_l},
              index=validated.index).div(validated.n_sents, axis='index')
all_methods = all_methods.apply(lambda x: map(normalize, x))
all_methods['vader'] = validated['vader']
all_methods['A_Score'] = validated['A_Score']
all_methods.corr()
all_methods = all_methods.apply(f)

for method in all_methods.columns[:-1]:
    print "Print classification metrics for method %s ...... " %method
    print classification_report(all_methods['A_Score'], all_methods[method])
