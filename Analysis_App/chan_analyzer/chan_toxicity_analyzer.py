import argparse
import json
import psycopg2
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import datetime

# Initialize Spacy for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 3000000  # Increase the limit to 3 million characters

DB_DETAILS = {
        "host":"localhost",
        "database":"chan_crawler",
        "user":"postgres",
        "password":"testpassword",
        "port":"5433"
}

def fetch_data(query, params=None):
    """Fetch data from the database."""
    connection = psycopg2.connect(**DB_DETAILS)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]

def fetch_snt_posts(date_range, country_name):
    """Fetch posts for sentiment analysis based on date range and country."""
    if date_range == "ALL":
        query = """
        SELECT data FROM posts
        WHERE toxicity_class = 'flag' AND data->>'country_name' = %s;
        """
        params = (country_name,)
    else:
        start_date, end_date = date_range.split("-")
        start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        query = """
        SELECT data FROM posts
        WHERE toxicity_class = 'flag' AND data->>'country_name' = %s
        AND to_date((data->>'now')::text, 'MM/DD/YY') BETWEEN %s AND %s;
        """
        params = (country_name, start_date, end_date)

    rows = fetch_data(query, params)

    return [row['com'] for row in rows if 'com' in row]


def fetch_stats_data(date_range):
    """Fetch counts of toxic and total posts for stats."""
    if date_range == "ALL":
        toxic_query = "SELECT COUNT(*) FROM posts WHERE toxicity_class = 'flag';"
        total_query = "SELECT COUNT(*) FROM posts;"
        toxic_count = fetch_data(toxic_query)[0]
        total_count = fetch_data(total_query)[0]
    else:
        start_date, end_date = date_range.split("-")
        start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        toxic_query = """
        SELECT COUNT(*) FROM posts
        WHERE toxicity_class = 'flag' AND to_date((data->>'now')::text, 'MM/DD/YY') BETWEEN %s AND %s;
        """
        total_query = """
        SELECT COUNT(*) FROM posts
        WHERE to_date((data->>'now')::text, 'MM/DD/YY') BETWEEN %s AND %s;
        """
        toxic_count = fetch_data(toxic_query, (start_date, end_date))[0]
        total_count = fetch_data(total_query, (start_date, end_date))[0]
    return toxic_count, total_count

def perform_sentiment_analysis(posts, output_file):
    """Perform sentiment analysis on posts and save results."""
    results = []
    for post in posts:
        analysis = TextBlob(post)
        results.append({
            "post": post,
            "polarity": analysis.polarity,
            "subjectivity": analysis.subjectivity
        })

    with open(output_file, "w") as f:
        for result in results:
            f.write(f"Post: {result['post']}\n")
            f.write(f"Polarity: {result['polarity']}\n")
            f.write(f"Subjectivity: {result['subjectivity']}\n\n")

    print(f"Sentiment analysis results written to {output_file}")

def calculate_and_print_stats(date_range):
    """Calculate and print statistics for toxic posts."""
    toxic_count, total_count = fetch_stats_data(date_range)
    print(f"Toxic comments: {toxic_count}")
    print(f"Total comments: {total_count}")


def fetch_posts_by_date(date_range, connection):
    """Fetch posts by date range and filter by toxicity_class='flag'."""
    start_date, end_date = date_range.split("-")
    start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d")

    query = """
    SELECT data FROM posts
    WHERE toxicity_class = 'flag'
    AND to_date((data->>'now')::text, 'MM/DD/YY') BETWEEN %s AND %s;
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()
    return [row[0]['com'] for row in rows if 'com' in row[0]]


def fetch_toxic_posts(connection):
    query = """
    SELECT data FROM posts
    WHERE toxicity_class = 'flag';
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
    return [row[0]['com'] for row in rows if 'com' in row[0]]

def perform_ner(posts, output_file):
    combined_text = " ".join(posts)  
    doc = nlp(combined_text.strip()) 

    # Group entities by category
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


    with open(output_file, "w") as f:
        f.write(html_content)

def perform_lda(posts, output_file, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    post_matrix = vectorizer.fit_transform(posts)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(post_matrix)

    # Extract topics and their top words
    topics = {}
    for idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topics[f"Topic {idx + 1}"] = top_words


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


    with open(output_file, "w") as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="4chan Toxicity Analyzer")
    parser.add_argument("--analysis-type", required=True, choices=["NER", "LDA", "SNT", "stats"], help="Type of analysis to perform.")
    parser.add_argument("--date-range", required=True, help="Date range in the format YYYYMMDD-YYYYMMDD or 'ALL'.")
    parser.add_argument("--country-name", help="Country name for sentiment analysis (required for SNT).")
    parser.add_argument("--output", help="Output file for results (required for NER, LDA, SNT).")


    args = parser.parse_args()

    if args.analysis_type == "SNT" and not args.country_name:
        raise ValueError("--country-name is required for sentiment analysis.")
    if args.analysis_type in ["NER", "LDA", "SNT"] and not args.output:
        raise ValueError("--output is required for NER, LDA, and SNT.")

    connection = psycopg2.connect(
        host="localhost",
        database="chan_crawler",
        user="postgres",
        password="testpassword",
        port="5433"
    )

    try:
        if args.date_range == "ALL":
            posts = fetch_toxic_posts(connection)
        else:
            posts = fetch_posts_by_date(args.date_range, connection)

        if args.analysis_type == "NER":
            perform_ner(posts, args.output)
            print(f"NER results written to {args.output}")
        elif args.analysis_type == "LDA":
            perform_lda(posts, args.output)
            print(f"LDA results written to {args.output}")
        elif args.analysis_type == "SNT":
            posts = fetch_snt_posts(args.date_range, args.country_name)
            perform_sentiment_analysis(posts, args.output)
        elif args.analysis_type == "stats":
            calculate_and_print_stats(args.date_range)
    finally:
        connection.close()

if __name__ == "__main__":
    main()
