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


df = pd.DataFrame({'price':[0,1,2], 'description': ['red', 'blue', 'green']})

d = [{'a': 1, 'b': 2, 'c': 3},{'a': 4, 'b': 5, 'c': 6},{'a': 7, 'b': 8, 'c': 9}]

bigdata = pd.concat([df, df2], ignore_index=False, axis = 1)