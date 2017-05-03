# ESSnet WP7 - Population #

The Office for National Statistics (ONS) is involved in the Big Data ESSNet. This is a programme of work funded by Eurostat, coordinated by CBS Netherlands and involving other 20 NSI partners. 

The aim of the Big Data ESSNet is to investigate how a combination of Big Data sources and existing official statistical data can be used to improve current statistics and create new statistics in statistical domains.
There are 9 work packages. Work Package 7 (WP7) focuses on the statistical domains of Population, Tourism/border crossings and Agriculture. 

From these domains, ONS has conducted a short pilot study in the "Population" use case.

### Description ###

The purpose of the case study is to examine the level of daily satisfaction by analysing the content of messages for the presence of defined expressions describing emotional states, e.g. joy, sadness, fear, anger. 

Additionally, emotional states of people can be analysed with respect to public events.

The idea is to explore how we might produce statistics on social sentiment from news sites/blogs/social media towards events/topics and how those can be linked to existing official statistics which annually measure population well-being.

This pilot study was set to provide:

* An in-depth exploration for the chosen source(s)
* An assessment of the API / web scraping conditions of the potential sources
* An exploration and application of lexicon-based sentiment analysis techniques
* An analysis of the results produced between the data sources and within different time units considered
* An assessment of the Facebook users leaving comments to determine distribution of gender and residency, and how this relates to the Guardian readership

### Content ###

The *WP7_ONS_files* folder contains the following files and folders:

##### Report

* **Final_doc.docx** the final report document

##### Python scripts

* **collect_data.py** to collect data (comments and posts) from the Facebook API
* **get_sentiment.py** to compute the sentiment using the different lexicons. The files generated are then saved in the **data** folder 
* **analysis.py** for reproducing the analysis and the charts of the report
* **utils.py** containing some util functions

##### Folders

* **images** folder which contains all images used in the final document and generated with the **analysis.py** file
* **data** folder which contains the processed data for the analysis. 
    - **comments.csv** the processed comments collected through the Facebook API excluding empty comments
    - **posts.csv** the processed posts collected through the Facebook API
    - **sentiment.csv** the normalised sentiment scores obtained with the Vader, NRC, Syuzhet, AFinn and Bing lexicons
    - **emotions.csv** the emotions scores calculated using the NRC lexicon
* **lexicons** folder which contains the lexicons used in the analysis, i.e. NRC, Syuzhet, AFinn and Bing in csv format