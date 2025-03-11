import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Database connection parameters
db_params = {
    'dbname': 'reddit_crawler',
    'user': 'postgres',
    'password': 'testpassword',
    'host': 'localhost',
    'port': '5432'
}

# Query to fetch required data
query = """
SELECT toxicity_class, COUNT(*) AS comment_count
FROM subreddit_posts_comment
WHERE comment_date BETWEEN '2024-11-06 05:30:00' AND '2024-11-06 14:30:00'
GROUP BY toxicity_class;
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
print(df)

# Ensure both "flag" and "normal" classes are represented
data = {'toxicity_class': ['flag', 'normal'], 'comment_count': [0, 0]}
df_full = pd.DataFrame(data).merge(df, on='toxicity_class', how='left').fillna(0)
df_full['comment_count'] = df_full['comment_count_x'] + df_full['comment_count_y']

# Define data
non_toxic_count = df_full.loc[df_full['toxicity_class'] == 'normal', 'comment_count'].values[0]
toxic_count = df_full.loc[df_full['toxicity_class'] == 'flag', 'comment_count'].values[0]

# Plotting the summarized stacked bar chart
plt.figure(figsize=(8, 6))

# Create the bar chart
plt.bar(['Comments'], [non_toxic_count], label='Non-Toxic (normal)', color='green')
plt.bar(['Comments'], [toxic_count], bottom=[non_toxic_count], label='Toxic (flag)', color='red')

# Add annotations
plt.text(0, non_toxic_count / 2, str(int(non_toxic_count)), ha='center', va='center', color='white', fontsize=12)
plt.text(0, non_toxic_count + toxic_count / 2, str(int(toxic_count)), ha='center', va='center', color='white', fontsize=12)

# Add labels and legend
plt.title('Summarized Stacked Bar Chart of Toxic and Non-Toxic Comments')
plt.ylabel('Number of Comments')
plt.legend()

# Save the plot as an image
plt.tight_layout()
plt.savefig('stacked_bar_chart.png')  # Saves the plot as a PNG file
print("Plot saved as 'stacked_bar_chart.png'")

