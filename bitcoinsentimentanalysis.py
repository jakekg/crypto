import requests
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

def fetch_news(coin='bitcoin'):
    url = f'https://newsapi.org/v2/everything?q={coin}&apiKey=XXXXXXXXXXXXXXXXXXXXX'  
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()['articles']
        return articles
    else:
        print(f"Failed to fetch news: {response.status_code}")
        print(response.text)
        return []

news_articles = fetch_news()

if news_articles:
    for article in news_articles:
        title = article['title']
        published_at = article['publishedAt']
        print(f"Title: {title}")
        print(f"Published At: {published_at}")
        print()
else:
    print("No news articles fetched.")

def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        text = article['title']
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        sentiments.append((text, sentiment))
    return sentiments

sentiment_results = analyze_sentiment(news_articles)

numeric_sentiments = [(article, sentiment) for article, sentiment in sentiment_results if isinstance(sentiment, (int, float))]

for article, sentiment in numeric_sentiments:
    print(f"Article: {article}")
    print(f"Sentiment: {sentiment}\n")

if numeric_sentiments:
    sentiment_df = pd.DataFrame(numeric_sentiments, columns=['Article', 'Sentiment'])
    sentiment_df['Sentiment'].plot(kind='hist', bins=20)
    plt.title('Sentiment Analysis of Bitcoin News Articles, May 2024')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.text(1.05, -10, 'Source: News API', ha='right', fontsize=10)
    plt.show()
else:
    print("No numeric sentiment data to plot.")