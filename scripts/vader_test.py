from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


my_example_text = """I begin this story with a neutral statement.
Basically this is a very silly test.
You are testing the Syuzhet package using short, inane sentences.
I am actually very happy today.
I have finally finished writing this package.
Tomorrow I will be very sad.
I won't have anything left to do.
I might get angry and decide to do something horrible.
I might destroy the entire package and start from scratch.
Then again, I might find it satisfying to have completed my first R package.
Honestly this use of the Fourier transformation is really quite elegant.
You might even say it's beautiful!"""
s_v = tokenize.sent_tokenize(my_example_text)
assert(len(s_v) == 12)

analyzer = SentimentIntensityAnalyzer()

# The 'compound' score is computed by summing the valence scores of each
# word in the lexicon, adjusted according to the rules, and then normalized
# to be between -1 (most extreme negative) and +1 (most extreme positive).
# This is the most useful metric if you want a single unidimensional
# measure of sentiment for a given sentence.
# Calling it a 'normalized, weighted composite score' is accurate.

# The 'pos', 'neu', and 'neg' scores are ratios for proportions of text
# that fall in each category (so these should all add up to be 1...
# or close to it with float operation).  These are the most useful metrics
# if you want multidimensional measures of sentiment for a given sentence.

vader_vector = []
for sentence in s_v:
    vs = analyzer.polarity_scores(sentence)
    vader_vector.append(vs['compound'])
