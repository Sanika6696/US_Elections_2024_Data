import argparse
import psycopg2
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import datetime

# Initialize Spacy for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3000000  # Increase the limit to 3 million characters

# Database connection details
DB_DETAILS = {
    "host": "localhost",
    "database": "reddit_crawler",
    "user": "postgres",
    "password": "testpassword",
    "port": 5433
}

def fetch_data(query, params=None):
    """Fetch data from the database based on the provided query."""
    connection = psycopg2.connect(**DB_DETAILS)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]


def perform_ner(comments, output_file):
    """Perform Named Entity Recognition and save results as HTML with collapsible dropdowns."""
    combined_text = " ".join(comments)
    doc = nlp(combined_text)
    entity_dict = {}
    for ent in doc.ents:
        if ent.label_ not in entity_dict:
            entity_dict[ent.label_] = set()
        entity_dict[ent.label_].add(ent.text)

    # Generate HTML with collapsible dropdowns
    html_content = """
    <html>
    <head>
    <style>
        .dropdown {
            margin-bottom: 10px;
        }
        .dropdown-content {
            display: none;
            margin-left: 20px;
        }
        .dropdown-button {
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px;
            text-align: left;
            width: 100%;
        }
    </style>
    <script>
        function toggleDropdown(id) {
            const content = document.getElementById(id);
            if (content.style.display === "none") {
                content.style.display = "block";
            } else {
                content.style.display = "none";
            }
        }
    </script>
    </head>
    <body>
    <h1>Named Entity Recognition Results</h1>
    """

    for category, entities in entity_dict.items():
        html_content += f"""
        <div class="dropdown">
            <button class="dropdown-button" onclick="toggleDropdown('{category}')">{category}</button>
            <div id="{category}" class="dropdown-content">
                <ul>
        """
        for entity in sorted(entities):
            html_content += f"<li>{entity}</li>"
        html_content += """
                </ul>
            </div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # Write the HTML content to the output file
    with open(output_file, "w") as f:
        f.write(html_content)



def perform_lda(comments, output_file):
    """Perform Latent Dirichlet Allocation and save topics as HTML."""
    vectorizer = CountVectorizer(stop_words='english')
    data_matrix = vectorizer.fit_transform(comments)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(data_matrix)

    # Extract topics and their top words
    topics = {}
    for idx, topic in enumerate(lda.components_):
        topics[f"Topic {idx + 1}"] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]

    # Generate HTML output
    html_content = """
    <html>
    <head>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .topic {
            margin-bottom: 20px;
        }
    </style>
    </head>
    <body>
    <h1>Latent Dirichlet Allocation Topics</h1>
    """

    for topic, words in topics.items():
        html_content += f"""
        <div class="topic">
            <h2>{topic}</h2>
            <p>{', '.join(words)}</p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # Write the HTML content to the output file
    with open(output_file, "w") as f:
        f.write(html_content)


def perform_sentiment_analysis(comments, output_file):
    """Perform sentiment analysis on comments and save results as a text file."""
    sentiment_results = []
    for comment in comments:
        analysis = TextBlob(comment)
        sentiment_results.append({"comment": comment, "polarity": analysis.polarity, "subjectivity": analysis.subjectivity})
    with open(output_file, "w") as f:
        for result in sentiment_results:
            f.write(f"Comment: {result['comment']}\nPolarity: {result['polarity']}\nSubjectivity: {result['subjectivity']}\n\n")

def calculate_stats(date_range):
    """Calculate and print statistics for toxic comments."""
    if date_range == "ALL":
        toxic_query = "SELECT COUNT(*) FROM subreddit_posts_comment WHERE toxicity_class = 'flag';"
        total_query = "SELECT COUNT(*) FROM subreddit_posts_comment;"
    else:
        start_date, end_date = date_range.split("-")
        toxic_query = """
        SELECT COUNT(*) FROM subreddit_posts_comment
        WHERE toxicity_class = 'flag' AND comment_date BETWEEN %s AND %s;
        """
        total_query = """
        SELECT COUNT(*) FROM subreddit_posts_comment
        WHERE comment_date BETWEEN %s AND %s;
        """
    toxic_count = fetch_data(toxic_query, (start_date, end_date) if date_range != "ALL" else None)[0]
    total_count = fetch_data(total_query, (start_date, end_date) if date_range != "ALL" else None)[0]
    print(f"Toxic comments: {toxic_count}")
    print(f"Total comments: {total_count}")

def main():
    parser = argparse.ArgumentParser(description="Reddit Toxicity Analyzer")
    parser.add_argument("--analysis-type", required=True, choices=["NER", "LDA", "SNT", "stats"], help="Type of analysis to perform.")
    parser.add_argument("--date-range", required=True, help="Date range in the format YYYYMMDD-YYYYMMDD or 'ALL'.")
    parser.add_argument("--upvote-threshold", type=int, help="Minimum upvotes for sentiment analysis.")
    parser.add_argument("--output", required=True, help="Output file for results.")

    args = parser.parse_args()

    if args.date_range != "ALL":
        start_date = datetime.datetime.strptime(args.date_range.split("-")[0], "%Y%m%d")
        end_date = datetime.datetime.strptime(args.date_range.split("-")[1], "%Y%m%d")
    else:
        start_date = end_date = None

    if args.analysis_type == "NER":
        query = """
        SELECT body FROM subreddit_posts_comment
        WHERE toxicity_class = 'flag' {};
        """.format("AND comment_date BETWEEN %s AND %s" if start_date and end_date else "")
        comments = fetch_data(query, (start_date, end_date) if start_date and end_date else None)
        perform_ner(comments, args.output)
    elif args.analysis_type == "LDA":
        query = """
        SELECT body FROM subreddit_posts_comment
        WHERE toxicity_class = 'flag' {};
        """.format("AND comment_date BETWEEN %s AND %s" if start_date and end_date else "")
        comments = fetch_data(query, (start_date, end_date) if start_date and end_date else None)
        perform_lda(comments, args.output)
    elif args.analysis_type == "SNT":
        if not args.upvote_threshold:
            raise ValueError("--upvote-threshold is required for sentiment analysis.")
        query = """
        SELECT body FROM subreddit_posts_comment
        WHERE ups >= %s {};
        """.format("AND comment_date BETWEEN %s AND %s" if start_date and end_date else "")
        comments = fetch_data(query, (args.upvote_threshold, start_date, end_date) if start_date and end_date else (args.upvote_threshold,))
        perform_sentiment_analysis(comments, args.output)
    elif args.analysis_type == "stats":
        calculate_stats(args.date_range)

if __name__ == "__main__":
    main()
