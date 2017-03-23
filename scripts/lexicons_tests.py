from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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

def rescale(x):
    return  2 * (x - min(x))/(max(x) - min(x)) - 1

def get_scores(text, vocab):
    vectorizer = CountVectorizer(vocabulary=vocab['word'])
    vect = vectorizer.fit_transform(text)
    values = np.array(vocab['value'])
    scores  = vect.dot(values)
    return scores

def get_nrc_emotions(text):
    vectorizer = CountVectorizer(vocabulary=nrc['word'])
    vect = vectorizer.fit_transform(text)
    colnames = [u'anger', u'anticipation', u'disgust',
                u'fear', u'joy', u'sadness', u'surprise',
                u'trust']
    values = csr_matrix(nrc[colnames].values)
    scores = vect.dot(values)
    return pd.DataFrame(scores.todense(), columns=colnames)


bing_scores = get_scores(df['message'], bing)
afinn_scores = get_scores(df['message'], afinn)
syuzhet_scores = get_scores(df['message'], syuzhet)
nrc_scores = get_scores(df['message'], nrc)

rescaled = pd.DataFrame({'bing': bing_scores,
              'afinn': afinn_scores,
              'syuzhet': syuzhet_scores,
              'nrc': nrc_scores},
              index=pd.to_datetime(df['created_time'])).apply(rescale)

rescaled.plot()

emotions = get_nrc_emotions(df['message'])

emotions.set_index(pd.to_datetime(df['created_time'])).plot()
emotions.set_index(pd.to_datetime(df['created_time'])).resample('H').mean().plot()

df['created_time']=pd.to_datetime(df['created_time'])
df.set_index(pd.to_datetime(df.index), inplace=True)
df.resample('H').count()['comment_id'].plot(legend=True, label='Volume')

# Correlation matrix between methods
rescaled.corr()

rescaled.resample('.5*H').mean().plot()
