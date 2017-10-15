### Sentiment Analysis of Tweets for Movie Kingsman: The Golden Circle

The objective of analysis is to explore text analytics by analyzing sentiment of tweets for the movie Kingsman: The Golden Circle.

1_tweets_search

By using Twython, downloaded 10,000 English only tweets from Twitter with keywords **_kingsman_ or _kingsmanmovie_** excluding any retweets. Tweets were created on September 23, 2017, one day after release of the movie in US. All tweets are stored to **_tweets_kingsman.txt_** in JSON format for further analysis.

2_tweets_analysis

* remove any **_@user_**, links, numbers and stopwords to keep letters only in lowercase.
* determine sentiment of each cleaned text, **_Positive, Neutral, or Negative_**. In the pie chart below, the percentage of positive, neutral and negative tweets are about 61%, 29% and 10%, which initially indicates that the movie has been received well by the audience.

![alt text](https://github.com/DHuangHRB/sideproject/blob/master/sentiment_analysis_of_tweets/images/piechart.png)

* visualize words of positive and negative tweets with word cloud to display the most frequent words respectively.
![alt text](https://github.com/DHuangHRB/sideproject/blob/master/sentiment_analysis_of_tweets/images/pos_wc.png "Positive") 
![alt text](https://github.com/DHuangHRB/sideproject/blob/master/sentiment_analysis_of_tweets/images/neg_wc.png "Negative")

* deep dive with a much wider range of sentiments based on NRC sentiment dictionary including 10 emotion types **_anger, anticipation, disgust, fear, joy, sadness, surprise, trust, negative, positive_**. In the bar chart of these emotions, the positive sentiments(positive, joy, trust) have higher scores than negative sentiments(negative, anger, fear)
![alt text](https://github.com/DHuangHRB/sideproject/blob/master/sentiment_analysis_of_tweets/images/nrcchart.png) 

The approaches above seem to suggest that the move Kingsman: The Golden Circle has been well received by the audience, as measured by positive, negative sentiments; as well as by the various emotions. It should be noted that the analysis is only based on a sample of 10,000 tweets created on one day after movie release. There may be more advanced methods to do the similar analysis.
