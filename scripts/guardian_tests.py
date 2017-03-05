
import requests


import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime, timedelta


url = "https://www.theguardian.com/discussion/p/625q2?page=1"

response = requests.get(url)
soup = BeautifulSoup(response.text, "lxml")

comments = soup.findAll('li', attrs = {'itemtype': 'http://schema.org/Comment'}) 
title = soup.find('h1', attrs = {'itemprop': 'headline'})
article = title.find('a')['href']

def get_data(comment):
    
    obj = dict()
    obj['article'] = article
    comment_id = comment['data-comment-id']
    author = comment['data-comment-author']
    timestamp = comment['data-comment-timestamp']
    author_id = comment['data-comment-author-id']
    reply = comment.find('a', attrs = {'class': 'js-discussion-author-link'})
    if reply:
        reply_to = reply['href'].split('-')[1]
        obj['reply_to'] = reply_to
    text = comment.find('div', attrs = {'class': 'd-comment__body'}).get_text()
    upvotes = comment.find('span', attrs = {'class': 'd-comment__recommend-count--old'})
    upvotes = int(upvotes.get_text())
    
    obj.update({
        'comment_id': comment_id,
        'author': author,
        'timestamp': timestamp,
        'author_id': author_id,
        'text': text,
        'upvotes': upvotes
    })
    
    return obj
    
comments = [get_data(c) for c in comments]

url = "https://www.theguardian.com/politics/2017/feb/25/brexit-fintech-exodus-begins-london-eu-luxembourg"
response = requests.get(url)
soup = BeautifulSoup(response.text, "lxml")

soup.find('div', attrs = {'id': 'comments'})['data-discussion-key']
link = 'https://www.theguardian.com/uk-news/2016/dec/01/london-mayor-issues-pollution-warnings-at-bus-stops-and-tube-stations'

response = requests.get(link)
soup = BeautifulSoup(response.text, "lxml")

[author.text.strip() for author in soup.findAll('span', attrs={'itemprop': 'author'})]

import time
from selenium import webdriver
url = "https://www.theguardian.com/uk-news/2016/dec/01/all"
driver = webdriver.Chrome("/Users/Alessandra/Documents/WebDriver/chromedriver")
driver.get(url)
time.sleep(2)

articles = driver.find_elements_by_xpath("//div[@class='fc-item__container']")
for article in articles:
    title = article.find_element_by_xpath(".//div[@class='fc-item__header']")
    title_text = title.text
    title_url = title.find_element_by_xpath(".//a").get_attribute("href")
    try:
        comments_count = article.find_element_by_xpath(".//a[@data-link-name='Comment count']").text
    except:
        comments_count = '0'
        
    comments_count = int(comments_count.replace(',', ''))
    
    obj = {
        'article_title': title_text,
        'article_url': title_url,
        'comments_count': comments_count
    }
    print obj
driver.close()


import pandas as pd
fb = pd.read_json("theguardian-posts-fb.json")
    
fb['likes'] = fb.likes.apply(lambda x: x['summary']['total_count'])
fb['shares'] = fb.shares.apply(lambda x: x['count'])
fb.set_index('created_time', inplace = True)
    
by_day = fb.resample('D').count()

by_day['id'].plot(ylim=(0,30), legend = True, label = 'Post per day')

fb = pd.read_json("theguardian-posts-fb-2.json")
    
fb['likes'] = fb.likes.apply(lambda x: x['summary']['total_count'])
fb['comments'] = fb.comments.apply(lambda x: x['summary']['total_count'])
fb['shares'] = fb.shares.apply(lambda x: x['count'] if isinstance(x, dict) else np.nan)
fb.set_index('created_time', inplace = True)
    
by_day = fb.resample('D').count()

by_day['id'].plot(legend = True, label = 'Post per day')


import re
from urlparse import urlparse

def get_extra(row):
    url = row['link']
    if not urlparse(url).netloc == 'www.theguardian.com':
        return pd.Series({'tags':None, 'article_title':None,
                      'authors': None, 'categories': None,
                      'main_category' : None})
    
    response = requests.get(url)
    time.sleep(2)
    print "Response received from %s" %url
    soup = BeautifulSoup(response.text, "lxml")
    tags = [tag.text.strip() for tag in soup.findAll('a', attrs={'class': 'submeta__link'})]
    article_title = soup.find(attrs={'itemprop': 'headline'}).text.strip()
    authors = [author.text.strip() for author in soup.findAll('span', attrs={'itemprop': 'author'})]
    categories = {category.text.strip().lower() for category in soup.findAll('a', attrs={'class': 'signposting__action'})}
    main_category = re.search(re.compile(r'theguardian\.com\/([\w-]*)'), url).group(1)
    return pd.Series({'tags':tags, 'article_title':article_title,
                      'authors': authors, 'categories': categories,
                      'main_category' : main_category})
                      

extra = fb.apply(get_extra, axis = 1)

final = fb.merge(extra, left_index=True, right_index=True)

by_all = final.groupby('main_category').agg({'likes':{'likes_mean':'mean', 'likes_count':'sum'},
                                               'comments':{'comments_mean':'mean', 'comments_count':'sum'},
                                               'shares':{'shares_mean':'mean', 'shares_count':'sum'}
                                                })

by_all.columns = by_all.columns.droplevel(0)

by_all[['likes_count', 'comments_count', 'shares_count']].plot(kind='barh', legend = True)
by_all['comments_count'].sort_values().plot(kind='barh', legend = True)
by_all['comments_mean'].sort_values().plot(kind='barh', legend = True)

main_categories = set(final.main_category) - {None}

cat_tags = dict()
for category in main_categories:
    subset = final[final['main_category'] == category]
    tags = {x for y in subset['tags'] for x in y}
    cat_tags[category] = ','.join(tags)
    
df = d_to_df(cat_tags)
df.to_csv('cat_tags.csv',encoding='utf-8')


sub_cat = dict()
for category in main_categories:
    subset = final[final['main_category'] == category]
    cats = {x for y in subset['categories'] for x in y}
    sub_cat[category] = ','.join(cats)
    
df = d_to_df(sub_cat)
df.to_csv('sub_cat.csv',encoding='utf-8')

import matplotlib.pyplot as plt
import seaborn; seaborn.set()
seaborn.set_style("whitegrid")
g = seaborn.FacetGrid(final, col="main_category", size = 4, col_wrap = 4)
g.map(plt.hist, "comments", alpha = .4);
