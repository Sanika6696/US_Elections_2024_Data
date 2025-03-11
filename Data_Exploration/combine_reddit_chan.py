import pandas as pd
import matplotlib.pyplot as plt
import json
import re
from collections import defaultdict
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
default_stop_words = set(stopwords.words('english'))

custom_stop_words = {"anonymous", "br", "gt", "replies", "href", "quotelink", "it's", "don't", "he's", "got", "think", "everyone", "get", "every", "getting", 
                     "people", "actually", "tell", "you're", "take", "going", "one", "even", "you039re", "he039s", "hrefp486587433", "it039s", "doesn039t", 
                     "don039t", "called", "can039t", "they039re", "hrefp486590420", "would", "know", "sure", "youre", "time", "day", "span", "like", "dont", 
                     "really", "see", "could", "hrefp486588990", "hrefp46589694", "still", "didn039t", "said", "also", "removed", "doesnt", "something", "didnt", "i039m",
                     "nothing", "another", "cant", "say", "good", "many", "want"
                     }

stop_words = default_stop_words.union(custom_stop_words)

# Load datasets
#chan_data = pd.read_csv("./chan_posts_1_3_nov.csv") 
chan_data = pd.read_csv("./chan_posts_1_14.csv") 
#reddit_comments_df = pd.read_csv("./subreddit_posts_comment_1_3.csv")  # Replace with your file
reddit_comments_df = pd.read_csv("./subreddit_posts_comment_1-14.csv")  # Replace with your file
#reddit_post_df = pd.read_csv("./subreddit_posts_1_3.csv")  # Replace with your file
reddit_post_df = pd.read_csv("./subreddit_posts_1-14.csv")  # Replace with your file

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Preprocess text function
def preprocess_text(text):
    text = str(text)
    text = re.sub(r"[^\w\s]", "", text.lower())  # Remove special characters and lowercase
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]  # Remove stopwords and short words
    return words

# 4chan threads: Extract and preprocess words
chan_data["comment"] = chan_data["data"].apply(lambda x: json.loads(x).get("com", "") if pd.notnull(x) else "")
chan_data["words"] = chan_data["comment"].apply(preprocess_text)
chan_words = [word for words in chan_data["words"] for word in words]
chan_word_counts = Counter(chan_words)

# Reddit comments: Extract and preprocess words
reddit_comments_df["words"] = reddit_comments_df["body"].apply(lambda x: preprocess_text(x) if pd.notnull(x) else [])
reddit_comment_words = [word for words in reddit_comments_df["words"] for word in words]
reddit_comment_word_counts = Counter(reddit_comment_words)

# Reddit posts: Extract and preprocess words
reddit_post_df["words"] = reddit_post_df["title"].apply(lambda x: preprocess_text(x) if pd.notnull(x) else [])
reddit_post_words = [word for words in reddit_post_df["words"] for word in words]
reddit_post_word_counts = Counter(reddit_post_words)

# Generate word clouds
chan_wordcloud = WordCloud(
    width=400, height=400,
    background_color="white",
    colormap="Blues",
    max_words=100
).generate_from_frequencies(chan_word_counts)

reddit_comment_wordcloud = WordCloud(
    width=400, height=400,
    background_color="white",
    colormap="Greens",
    max_words=100
).generate_from_frequencies(reddit_comment_word_counts)

reddit_post_wordcloud = WordCloud(
    width=400, height=400,
    background_color="white",
    colormap="Purples",
    max_words=100
).generate_from_frequencies(reddit_post_word_counts)

# Display the word clouds
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 4chan Word Cloud
axes[0].imshow(chan_wordcloud, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("4chan Threads", fontsize=14, fontweight="bold")

# Reddit Comments Word Cloud
axes[1].imshow(reddit_comment_wordcloud, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Reddit Comments", fontsize=14, fontweight="bold")

# Reddit Posts Word Cloud
axes[2].imshow(reddit_post_wordcloud, interpolation="bilinear")
axes[2].axis("off")
axes[2].set_title("Reddit Posts", fontsize=14, fontweight="bold")

# Adjust layout
plt.tight_layout()
#plt.show()
plt.savefig("./combine_world_cloud.png")
plt.close

