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
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap


# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
default_stop_words = set(stopwords.words('english'))

custom_stop_words = {"anonymous", "br", "gt", "replies", "href", "quotelink", "it's", "don't", "he's", "got", "think", "everyone", "get", "every", "getting", 
                     "people", "actually", "tell", "you're", "take", "going", "one", "even", "you039re", "he039s", "hrefp486587433", "it039s", "doesn039t", 
                     "don039t", "called", "can039t", "they039re", "hrefp486590420"}
stop_words = default_stop_words.union(custom_stop_words)

# Load datasets
#chan_data = pd.read_csv("chan_posts_1_3_nov.csv") 
chan_data = pd.read_csv("chan_posts_1_14.csv") 

# Function to extract fields from the 'data' JSON-like string
def extract_chan_data(row):
    try:
        # Parse the JSON-like string
        data_dict = json.loads(row)
        # Extract relevant fields
        return {
            "comment": data_dict.get("com", ""),  # Extract comment
            "date": datetime.strptime(data_dict.get("now", ""), "%m/%d/%y(%a)%H:%M:%S"),  # Extract and parse date
            "country": data_dict.get("country_name", "Unknown")  # Extract country
        }
    except Exception as e:
        return {"comment": None, "date": None, "country": None}  # Return None if parsing fails

# Apply the extraction function to each row of the `data` column
chan_data_extracted = chan_data["data"].apply(extract_chan_data).apply(pd.Series)

# Merge the extracted data into the original DataFrame
chan_data = pd.concat([chan_data, chan_data_extracted], axis=1)

# Perform sentiment analysis on the extracted `comment` field
analyzer = SentimentIntensityAnalyzer()
chan_data["sentiment"] = chan_data["comment"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"] if pd.notnull(x) else None
)

# Inspect the updated DataFrame
print(chan_data[["comment", "date", "country", "sentiment"]].head())

#********************************* Sentiment Analysis based on countries

# Group by country to analyze sentiment
country_sentiment = chan_data.groupby("country").agg(
    total_comments=("comment", "count"),
    avg_sentiment=("sentiment", "mean"),
    min_date=("date", "min"),
    max_date=("date", "max"),
).reset_index()

# Display sentiment grouped by country
print(country_sentiment)

# Plot sentiment by country

plt.figure(figsize=(14, 8))
sns.barplot(
    x="avg_sentiment",
    y="country",
    data=country_sentiment.sort_values(by="avg_sentiment", ascending=False),
    palette="coolwarm"
)

# Add labels and title
plt.title("Average Sentiment by Country (4chan)", fontsize=18, fontweight="bold", pad=20)
plt.xlabel("Average Sentiment Score", fontsize=14)
plt.ylabel("Country", fontsize=14)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
#plt.show()
plt.savefig("./sentiment_based_on_country.png")
plt.close

# ***************************Country Map

# # Group by country to calculate sentiment
# country_sentiment = chan_data.groupby("country").agg(
#     total_comments=("comment", "count"),
#     avg_sentiment=("sentiment", "mean")
# ).reset_index()

# # Load world shapefile
# world = gpd.read_file("E:/Classes/Social_Media_Data_Science_Pipeline/Database_backup/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# # Merge world GeoDataFrame with country_sentiment DataFrame
# world_sentiment = world.merge(country_sentiment, how="left", left_on="NAME", right_on="country")

# # Define a custom diverging colormap (Red for negative, Blue for positive)
# cmap = LinearSegmentedColormap.from_list("sentiment_cmap", ["darkred", "white", "darkblue"])

# # Plot the map
# fig, ax = plt.subplots(1, 1, figsize=(14, 8))
# world_sentiment.plot(
#     column="avg_sentiment",  # Column to base the color on
#     cmap=cmap,
#     legend=True,
#     legend_kwds={"label": "Average Sentiment", "orientation": "horizontal"},
#     ax=ax,
#     missing_kwds={"color": "lightgrey", "label": "No Data"}
# )

# # Customize the plot
# ax.set_title("World Map of Sentiment Analysis by Country (4chan)", fontsize=16, fontweight="bold")
# ax.set_axis_off()  # Turn off axis lines and labels

# # Show the plot
# plt.tight_layout()
# #plt.show()

# Group by country to calculate sentiment
country_sentiment = chan_data.groupby("country").agg(
    total_comments=("comment", "count"),
    avg_sentiment=("sentiment", "mean")
).reset_index()

# Load the world shapefile
world = gpd.read_file("./ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")  # Update the path

# Merge world data with sentiment data
world_sentiment = world.merge(country_sentiment, how="left", left_on="NAME", right_on="country")

# Define a more vibrant diverging colormap
cmap = LinearSegmentedColormap.from_list("sentiment_cmap", ["darkred", "white", "darkblue"])

# Plot the map with adjusted figure size and color intensity
fig, ax = plt.subplots(1, 1, figsize=(20, 10))  # Increase figure size for better visibility
world_sentiment.plot(
    column="avg_sentiment",  # Column to base the color on
    cmap=cmap,
    legend=True,
    legend_kwds={"label": "Sentiment Scale (Negative to Positive)", "orientation": "horizontal", "shrink": 0.7},
    ax=ax,
    missing_kwds={"color": "lightgrey", "label": "No Data"},
    linewidth=0.5,  # Adjust border line width
    edgecolor="black"  # Add black borders to countries
)

# Add country names for countries with data
for _, row in world_sentiment.iterrows():
    if pd.notnull(row["avg_sentiment"]):  # Only label countries with sentiment data
        ax.text(
            row.geometry.centroid.x,
            row.geometry.centroid.y,
            row["NAME"],  # Use the country name
            fontsize=4,
            ha="center",
            color="black"  # Color for text
        )

# Customize the plot
ax.set_title("World Map of Sentiment Analysis by Country (4chan)", fontsize=18, fontweight="bold", pad=20)
ax.set_axis_off()  # Turn off axis lines and labels

# Show the plot
plt.tight_layout()
#plt.show()
plt.savefig("./sentiment_on_country_map.png")
plt.close


# ****************************

# Define policy terms and candidates
policy_terms = [
    "healthcare", "security", "economy", "guns", "immigration",
    "public health", "national security", "terrorism", "inflation",
    "unemployment", "trade policy", "climate change", "abortion",
    "equal pay", "voting rights", "health", "care"
]
candidates = {
    "Trump": ["trump", "donald trump", "donald", "him"], "Harris": ["harris", "kamala harris", "kamala", "her"]
    }

# Initialize a dictionary to store term frequencies
term_frequencies = {candidate: {term: 0 for term in policy_terms} for candidate in candidates}

# Process the comments
for _, row in chan_data.iterrows():
    comment = str(row["comment"]).lower()
    for candidate, name_variations in candidates.items():
        if any(name in comment for name in name_variations):  # Check for any variation of the candidate's name
            for term in policy_terms:
                term_frequencies[candidate][term] += len(re.findall(rf"\b{term}\b", comment))

# Convert term frequencies to a DataFrame
freq_df = pd.DataFrame(term_frequencies).T
freq_df = freq_df.reset_index().rename(columns={"index": "Candidate"})

# Melt the DataFrame for easier plotting
freq_melted = freq_df.melt(id_vars="Candidate", var_name="Policy Term", value_name="Frequency")

# Create a clustered bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Bar width and positions
bar_width = 0.35
x = range(len(policy_terms))

# Plot bars for each candidate
for i, (candidate, _) in enumerate(candidates.items()):
    subset = freq_melted[freq_melted["Candidate"] == candidate]
    ax.bar(
        [pos + (i * bar_width) for pos in x],
        subset["Frequency"],
        bar_width,
        label=candidate,
        alpha=0.7,
        edgecolor="black"
    )

# Customize the plot
ax.set_title("Frequency of Policy Keyword Mentions by Candidate (4chan)", fontsize=16, fontweight="bold")
ax.set_xlabel("Policy Keywords", fontsize=14)
ax.set_ylabel("Frequency of Mentions", fontsize=14)
ax.set_xticks([pos + bar_width / 2 for pos in x])
ax.set_xticklabels(policy_terms, rotation=45, ha="right", fontsize=10)
ax.legend(title="Candidate", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout and display
plt.tight_layout()
#plt.show()
plt.savefig("./chan_policy_keywords.png")
plt.close


# *******************

# Extract 'comment' field from 'data' column
chan_data["comment"] = chan_data["data"].apply(lambda x: json.loads(x).get("com", "") if pd.notnull(x) else "")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to preprocess text and calculate sentiment
def preprocess_and_analyze(text):
    try:
        text = str(text)  # Convert to string if not already
        text = re.sub(r"[^\w\s]", "", text.lower())  # Remove special characters and lowercase
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]  # Remove stopwords and short words
        
        sentiment = analyzer.polarity_scores(" ".join(words))  # Sentiment analysis
        return {"words": words, "sentiment": sentiment["compound"]}  # Return results as a dict
    except Exception as e:
        return {"words": [], "sentiment": 0}  # Return empty on error

# Apply the preprocessing and sentiment analysis to the 4chan comments
chan_data["processed"] = chan_data["comment"].map(
    lambda x: preprocess_and_analyze(x) if pd.notnull(x) else {"words": [], "sentiment": 0}
)

# Extract words and sentiment from the processed data
chan_data["words"] = chan_data["processed"].apply(lambda x: x["words"])
chan_data["sentiment"] = chan_data["processed"].apply(lambda x: x["sentiment"])

# Filter toxic comments (sentiment < -0.5)
toxic_comments = chan_data[chan_data["sentiment"] < -0.5]["words"]

# Flatten the list of toxic words
all_words = [word for words in toxic_comments for word in words]

# Count the most common toxic words
word_counts = Counter(all_words)
print("Most Common Used Words:", word_counts.most_common(10))

# Create a word cloud for the most frequent toxic words
wordcloud = WordCloud(
    width=800, height=400,
    background_color="white",
    colormap="Reds",
    max_words=100
).generate_from_frequencies(word_counts)

# Display the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequently Used Words from 4chan Data", fontsize=16, fontweight="bold")
#plt.show()
plt.savefig("./chan_world_cloud.png")
plt.close



