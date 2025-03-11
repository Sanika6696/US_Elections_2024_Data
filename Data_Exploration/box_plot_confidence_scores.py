import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Database connection parameters from the screenshot
db_params = {
    'dbname': 'reddit_crawler',
    'user': 'postgres',
    'password': 'testpassword',
    'host': 'localhost',
    'port': '5432'
}

# Query to fetch required data
query = """
SELECT toxicity_class, confidence_score
FROM subreddit_posts_comment
WHERE comment_date BETWEEN '2024-11-06 05:30:00' AND '2024-11-06 14:30:00';
"""

# Fetch data using psycopg2
try:
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()  # Fetch all rows
    colnames = [desc[0] for desc in cur.description]  # Get column names
    df = pd.DataFrame(data, columns=colnames)  # Create DataFrame
    conn.close()
except Exception as e:
    print(f"Error connecting to the database: {e}")

# Debug: Check the fetched data
print(df.head())

# Filter only rows with toxicity_class as 'normal' or 'flag'
df_filtered = df[df['toxicity_class'].isin(['normal', 'flag'])]

# Plotting the Box Plot
plt.figure(figsize=(8, 6))

# Box plot with seaborn
sns.boxplot(data=df_filtered, x='toxicity_class', y='confidence_score', hue='toxicity_class', dodge=False, palette={'normal': 'green', 'flag': 'red'})

# Add titles and labels
plt.title('Box Plot of Confidence Scores by Toxicity Class')
plt.xlabel('Toxicity Class')
plt.ylabel('Confidence Score')

# Save or display the plot in the current directory
plt.tight_layout()
plt.savefig('box_plot_filtered_confidence_scores_2.png')
plt.show()

