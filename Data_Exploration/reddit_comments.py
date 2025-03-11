import pandas as pd
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import spacy
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import re
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")

# Read the CSV file
#csv_file = "./subreddit_posts_comment_1_3.csv" 
csv_file = "./subreddit_posts_comment_1-14.csv" 
reddit_comments_df = pd.read_csv(csv_file)

# Read the CSV file
# chan_csv_file = "E:/Classes/Social_Media_Data_Science_Pipeline/Database_backup/chan_DB/posts_1_3_nov.csv" 
# chan_df = pd.read_csv(chan_csv_file)

#reddit_post_csv_file = "./subreddit_posts_1_3.csv"
reddit_post_csv_file = "./subreddit_posts_1-14.csv"
reddit_post_df = pd.read_csv(reddit_post_csv_file)

# # Ensure comment_date is in datetime format
reddit_comments_df["comment_date"] = pd.to_datetime(reddit_comments_df["comment_date"], errors="coerce")

# # Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to preprocess text with Spacy and calculate sentiment
def preprocess_and_analyze(text):
    try:
        # Preprocess text with Spacy
        doc = nlp(text)
        clean_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        
        # Calculate sentiment with VADER
        sentiment = analyzer.polarity_scores(clean_text)
        return sentiment["compound"]  # Compound score: overall sentiment polarity
    except:
        return None

# Apply preprocessing and sentiment analysis to the body column
reddit_comments_df["sentiment"] = reddit_comments_df["body"].apply(lambda x: preprocess_and_analyze(x) if pd.notnull(x) else None)

print(reddit_comments_df[["body", "sentiment", "comment_date"]].head())


# Separate positive and negative sentiments
reddit_comments_df["positive_sentiment"] = reddit_comments_df["sentiment"].apply(lambda x: x if x > 0 else 0)
reddit_comments_df["negative_sentiment"] = reddit_comments_df["sentiment"].apply(lambda x: x if x < 0 else 0)
reddit_comments_df["neutral_sentiment"] = reddit_comments_df["sentiment"].apply(lambda x: x if x == 0 else 0)


# Group by date and calculate averages
daily_positive_sentiment = reddit_comments_df.groupby(reddit_comments_df["comment_date"].dt.date)["positive_sentiment"].mean()
daily_negative_sentiment = reddit_comments_df.groupby(reddit_comments_df["comment_date"].dt.date)["negative_sentiment"].mean()
daily_neutral_sentiment = reddit_comments_df.groupby(reddit_comments_df["comment_date"].dt.date)["neutral_sentiment"].mean()


# *************************Daily Sentiment Trends observed on a daily basis - First Chart

plt.figure(figsize=(14, 8))

# Positive sentiment
plt.plot(daily_positive_sentiment.index, daily_positive_sentiment.values, 
         label="Positive Sentiment", color="green", linewidth=3, linestyle="--", marker="o")
plt.fill_between(daily_positive_sentiment.index, daily_positive_sentiment.values, alpha=0.1, color="green")

# Negative sentiment
plt.plot(daily_negative_sentiment.index, daily_negative_sentiment.values, 
         label="Negative Sentiment", color="red", linewidth=3, linestyle=":", marker="x")
plt.fill_between(daily_negative_sentiment.index, daily_negative_sentiment.values, alpha=0.1, color="red")


# Title and labels
plt.title("Daily Sentiment Trends: Positive vs Negative Sentiment", fontsize=18, fontweight="bold", pad=20)
plt.xlabel("Date", fontsize=14, labelpad=10)
plt.ylabel("Average Sentiment Polarity", fontsize=14, labelpad=10)
plt.legend(fontsize=12, loc="upper right", title="Sentiment Type")

# Grid and formatting
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45, fontsize=10)

# Annotations
max_pos_date = daily_positive_sentiment.idxmax()
max_pos_value = daily_positive_sentiment.max()
plt.annotate(f"Max Positive: {max_pos_value:.2f}", xy=(max_pos_date, max_pos_value), 
             xytext=(max_pos_date, max_pos_value + 0.1), 
             arrowprops=dict(facecolor='green', arrowstyle="->"), fontsize=12)

max_neg_date = daily_negative_sentiment.idxmin()
max_neg_value = daily_negative_sentiment.min()
plt.annotate(f"Max Negative: {max_neg_value:.2f}", xy=(max_neg_date, max_neg_value), 
             xytext=(max_neg_date, max_neg_value - 0.1), 
             arrowprops=dict(facecolor='red', arrowstyle="->"), fontsize=12)

plt.tight_layout()
#plt.show()
plt.savefig("./daily_trends.png")
plt.close()


# ******Sentiment Bar chart - Timeline for per hour sentiment analysis done

# # Create hourly bins
reddit_comments_df["hour"] = reddit_comments_df["comment_date"].dt.floor("h")

# Aggregate the sentiment by hour
hourly_sentiment = reddit_comments_df.groupby("hour").agg(
    positive_sentiment=("sentiment", lambda x: x[x > 0].sum()),  # Sum of positive sentiments
    negative_sentiment=("sentiment", lambda x: x[x < 0].sum()),  # Sum of negative sentiments
    avg_sentiment=("sentiment", "mean")  # Average sentiment per hour for reference
).reset_index()

# Configure the colormap and normalization
cmap = sns.diverging_palette(250, 30, l=65, as_cmap=True)  # Diverging colormap
# norm = Normalize(vmin=-1, vmax=1)  # Normalize sentiment scores between -1 and 1

# Create the plot
fig, ax = plt.subplots(figsize=(16, 8))

# Plot positive sentiments
ax.bar(
    hourly_sentiment["hour"],
    hourly_sentiment["positive_sentiment"],
    color="navajowhite",
    alpha=0.7,
    width=0.03,
    edgecolor="black",
    label="Positive Sentiment"
)

# Plot negative sentiments
ax.bar(
    hourly_sentiment["hour"],
    hourly_sentiment["negative_sentiment"],
    color="mediumorchid",
    alpha=0.7,
    width=0.03,
    edgecolor="black",
    label="Negative Sentiment"
)

# Add a horizontal line at 0
ax.axhline(0, color="black", linestyle="--", linewidth=1)

# Add titles and labels
ax.set_title("Sentiment Analysis Timeline", fontsize=18, fontweight="bold", pad=20)
ax.set_xlabel("Comments per Hour", fontsize=14, labelpad=10)
ax.set_ylabel("Sentiment Score", fontsize=14, labelpad=10)
ax.tick_params(axis="x", rotation=45, labelsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Add legend
ax.legend(fontsize=12, loc="upper left")

# Adjust layout and show plot
plt.tight_layout()
#plt.show()
plt.savefig("./sentiment_timeline.png")
plt.close()

# # *******************Trump vs Harris - Research Q2 - policy terms already mentioned used with the candidate names - Bar Chart

# List of policy terms
policy_terms = [
    "healthcare", "security", "economy", "guns", "immigration",
    "public health", "national security", "terrorism", "inflation",
    "unemployment", "trade policy", "climate change", "abortion",
    "equal pay", "voting rights", "border", "tariffs"
]

candidates = {
    "Trump": ["trump", "donald trump", "donald", "him"], "Harris": ["harris", "kamala harris", "kamala", "her"]
    }
# Initialize a dictionary to store frequencies
term_frequencies = {candidate: {term: 0 for term in policy_terms} for candidate in candidates}

# Process the comment data
if "body" in reddit_comments_df.columns:  # Assuming comments are in 'body' column
    for _, row in reddit_comments_df.iterrows():
        text = str(row["body"]).lower()  # Ensure text is lowercase for case-insensitive matching
        
        # Check for candidate mentions and count policy terms
        for candidate in candidates:
            if candidate.lower() in text:  # Check if the candidate is mentioned
                for term in policy_terms:
                    term_frequencies[candidate][term] += len(re.findall(rf"\b{term}\b", text))  # Count whole-word matches

# Convert the term frequencies to a DataFrame for visualization
freq_df = pd.DataFrame(term_frequencies).T
freq_df = freq_df.reset_index().rename(columns={"index": "Candidate"})

# Melt the DataFrame for easier plotting
freq_melted = freq_df.melt(id_vars="Candidate", var_name="Policy Term", value_name="Frequency")

# Plot a clustered bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Bar width and positions
bar_width = 0.35
x = range(len(policy_terms))

# Plot bars for each candidate
for i, candidate in enumerate(candidates):
    subset = freq_melted[freq_melted["Candidate"] == candidate]
    ax.bar(
        [pos + (i * bar_width) for pos in x],
        subset["Frequency"],
        bar_width,
        label=candidate,
        alpha=0.8,
        edgecolor="black"
    )

# Customize the plot
ax.set_title("Frequency of Policy Keyword Mentions by Candidate", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Policy Keywords", fontsize=14)
ax.set_ylabel("Frequency of Mentions", fontsize=14)
ax.set_xticks([pos + bar_width / 2 for pos in x])
ax.set_xticklabels(policy_terms, rotation=45, ha="right", fontsize=12)
ax.legend(title="Candidate", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout and display
plt.tight_layout()
#plt.show()
plt.savefig("./Policy_keywords_mentioned.png")
plt.close()

# Plot bars for each candidate
# for i, candidate in enumerate(candidates):
#     subset = freq_melted[freq_melted["Candidate"] == candidate]
#     ax.bar(
#         [pos + (i * bar_width) for pos in x],
#         subset["Frequency"],
#         bar_width,
#         label=candidate,
#         alpha=0.7,
#         edgecolor="black"
#     )

# # Customize the plot
# ax.set_title("Frequency of Policy Term Mentions by Candidate", fontsize=16, fontweight="bold")
# ax.set_xlabel("Policy Terms", fontsize=14)
# ax.set_ylabel("Frequency of Mentions", fontsize=14)
# ax.set_xticks([pos + bar_width / 2 for pos in x])
# ax.set_xticklabels(policy_terms, rotation=45, ha="right", fontsize=10)
# ax.legend(title="Candidate", fontsize=12)
# ax.grid(axis="y", linestyle="--", alpha=0.7)

# # Adjust layout and display
# plt.tight_layout()
# #plt.show()

# # *******************Trump vs Harris - Most frequently mentioned terms with Trump and Harris - Histogram

# Download stopwords if not already downloaded
nltk.download("stopwords")

nltk.download("punkt")
nltk.download("punkt_tab")


# Preprocess text: Remove stopwords, tokenize, and clean
stop_words = set(stopwords.words("english"))

# Add custom stop words for each candidate
custom_stopwords_trump = {"trump", "donald", "don't", "would" ,"people" ,"really" ,"say" ,"actually" ,"dont", "said", "got", "know", "think", "gt", "like", "still", 
                          "im","could", "time", "back", "well", "never", "going", "good", "cant", "didnt", "even", "want", "see", "since", "hes", "better",
                          "last", "go", "every", "one", "also", "get", "question", "make", "much", "youre", "already", "ever", "talking", "come", "us", "far", "things", 
                          "look", "oh", "years", "trying", "thats", "new", "thats", "thing", "getting", "anything", "republican", "something", "trumps"}
custom_stopwords_harris = {"harris", "kamala", "don't", "cant" ,"know" ,"gt" ,"people" ,"dont" ,"also", "like", "youre", "really", "see", "think", "trying", "actually", 
                           "said", "anything", "didnt", "much", "good", "since", "hes", "years", "side", "sure", "would", "every", "never", "go", "time", "say"
                           "get", "one", "im", "ever", "points", "talking", "going", "might", "make", "blah", "done", "says", "thought", "speak", "called", "act", "bet", 
                           "lot", "ive", "get", "say", "got", "goes", "everyone", "knows", "republican", "worse", "fact", "well", "understand", "matter"}


# Combine default and custom stop words
trump_stop_words = stop_words.union(custom_stopwords_trump)
harris_stop_words = stop_words.union(custom_stopwords_harris)


def preprocess_text(text, candidate_stop_words):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize text
    # return [word for word in tokens if word not in stop_words and word.isalpha()]
    return [word for word in tokens if word not in candidate_stop_words and word.isalpha()]

# Separate comments by candidate and preprocess
trump_comments = reddit_comments_df[reddit_comments_df["body"].str.contains("trump", case=False, na=False)]
harris_comments = reddit_comments_df[reddit_comments_df["body"].str.contains("harris", case=False, na=False)]

trump_words = [word for comment in trump_comments["body"] for word in preprocess_text(comment, trump_stop_words)]
harris_words = [word for comment in harris_comments["body"] for word in preprocess_text(comment, harris_stop_words)]

# Count word frequencies
trump_word_counts = Counter(trump_words)
harris_word_counts = Counter(harris_words)

# Get the top 10 most frequent words for each candidate
top_trump_words = trump_word_counts.most_common(30)
top_harris_words = harris_word_counts.most_common(30)

# Convert to DataFrames for plotting
trump_df = pd.DataFrame(top_trump_words, columns=["Word", "Frequency"])
harris_df = pd.DataFrame(top_harris_words, columns=["Word", "Frequency"])

# Plot Trump's most frequent words
plt.figure(figsize=(12, 6))
sns.barplot(data=trump_df, x="Frequency", y="Word", palette="Blues_r")
plt.title("Top 30 Most Frequent Words Associated with Trump", fontsize=16, fontweight="bold")
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Word", fontsize=12)
plt.tight_layout()
#plt.show()
plt.savefig("./Trump_most_freq.png")
plt.close()

# Plot Harris's most frequent words
plt.figure(figsize=(12, 6))
sns.barplot(data=harris_df, x="Frequency", y="Word", palette="Oranges_r")
plt.title("Top 30 Most Frequent Words Associated with Harris", fontsize=16, fontweight="bold")
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Word", fontsize=12)
plt.tight_layout()
#plt.show()
plt.savefig("./Harris_most_freq.png")
plt.close()


# ****************Reddit Posts

# Convert `created_at` to datetime format
reddit_post_df["created_at"] = pd.to_datetime(reddit_post_df["created_at"], errors="coerce")

# Drop rows with invalid dates
reddit_post_df = reddit_post_df.dropna(subset=["created_at"])

# Aggregate engagements (likes and comments) by date
engagement_by_date = reddit_post_df.groupby(reddit_post_df["created_at"].dt.date).agg(
    total_likes=("likes", "sum"),
    total_comments=("no_of_comments", "sum")
).reset_index()

# Convert `created_at` back to datetime for plotting
engagement_by_date["created_at"] = pd.to_datetime(engagement_by_date["created_at"])

# Normalize the values for better visualization (optional)
engagement_by_date["normalized_likes"] = engagement_by_date["total_likes"] / engagement_by_date["total_likes"].max()
engagement_by_date["normalized_comments"] = engagement_by_date["total_comments"] / engagement_by_date["total_comments"].max()

# Plot Engagement Around Key Political Events
plt.figure(figsize=(14, 8))

# Stacked bar chart for likes and comments
bar_width = 0.4
x_positions = range(len(engagement_by_date["created_at"]))

plt.bar(
    x_positions,
    engagement_by_date["total_likes"],
    color="orange",
    label="Likes",
    width=bar_width,
    alpha=0.8,
    edgecolor="black"
)

plt.bar(
    x_positions,
    engagement_by_date["total_comments"],
    bottom=engagement_by_date["total_likes"],
    color="blue",
    label="Comments",
    width=bar_width,
    alpha=0.8,
    edgecolor="black"
)

# Highlight specific dates (e.g., key political events)
key_dates = ["2024-11-05","2024-11-06", "2024-11-07"]  # Example dates
for key_date in key_dates:
    if pd.to_datetime(key_date).date() in engagement_by_date["created_at"].dt.date.values:
        index = engagement_by_date["created_at"].dt.date.values.tolist().index(pd.to_datetime(key_date).date())
        plt.axvline(x=index, color="red", linestyle="--", label=f"Key Event: {key_date}")

# Add labels, title, and legend
plt.title("Engagement Around Key Political Events", fontsize=18, fontweight="bold", pad=20)
plt.xlabel("Date", fontsize=14, labelpad=10)
plt.ylabel("Number of Engagements", fontsize=14, labelpad=10)
plt.xticks(x_positions, engagement_by_date["created_at"].dt.strftime("%b %d"), rotation=45, fontsize=10)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout and show plot
plt.tight_layout()
#plt.show()
plt.savefig("./daily_trends.png")
plt.close()

# *********************Sentiments over subreddits

# Ensure dates are in datetime format
reddit_post_df["created_at"] = pd.to_datetime(reddit_post_df["created_at"], errors="coerce")
reddit_comments_df["comment_date"] = pd.to_datetime(reddit_comments_df["comment_date"], errors="coerce")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis on comments
def calculate_toxicity(text):
    try:
        sentiment = analyzer.polarity_scores(text)
        return -sentiment["compound"]  # Negative scores represent more toxicity
    except:
        return None

reddit_comments_df["toxicity_score"] = reddit_comments_df["body"].apply(lambda x: calculate_toxicity(x) if pd.notnull(x) else None)

merged_df = reddit_post_df.merge(reddit_comments_df, on="post_id", how="inner")

# Extract date and group by subreddit and date
merged_df["date"] = merged_df["comment_date"].dt.date
toxicity_trends = merged_df.groupby(["subreddit_name", "date"]).agg(
    avg_toxicity=("toxicity_score", "mean")  # Average toxicity per subreddit per day
).reset_index()

# Filter for top 5 subreddits by number of posts
top_subreddits = merged_df["subreddit_name"].value_counts().head(5).index
filtered_trends = toxicity_trends[toxicity_trends["subreddit_name"].isin(top_subreddits)]

# Plot toxicity trends
plt.figure(figsize=(14, 8))
for subreddit in top_subreddits:
    subreddit_data = filtered_trends[filtered_trends["subreddit_name"] == subreddit]
    plt.plot(subreddit_data["date"], subreddit_data["avg_toxicity"], label=subreddit, marker='o')

# Customize the plot
plt.title("Toxicity Trends Over Time by Subreddit", fontsize=18, fontweight="bold", pad=20)
plt.xlabel("Date", fontsize=14, labelpad=10)
plt.ylabel("Average Toxicity Score", fontsize=14, labelpad=10)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Subreddit", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout and show plot
plt.tight_layout()
#plt.show()
plt.savefig("./sentiments_subreddit.png")
plt.close()
