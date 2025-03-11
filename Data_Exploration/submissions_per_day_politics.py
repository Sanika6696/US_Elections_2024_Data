import pandas as pd
import plotly.express as px
import plotly.io as pio

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('subreddit_posts_202412031212.csv')

# Convert 'created_at' to a proper datetime format
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Filter the data to only include the r/politics subreddit
df_politics = df[df['subreddit_name'] == 'politics']

# Define the date range for filtering
start_date = '2024-11-01 00:00:00'
end_date = '2024-11-14 23:59:59'

# Convert the date strings to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter for the date range
df_politics = df_politics[(df_politics['created_at'] >= start_date) & (df_politics['created_at'] <= end_date)]

# Group by day and count the number of submissions
df_politics['date'] = df_politics['created_at'].dt.date  # Extract just the date
daily_counts = df_politics.groupby('date').size().reset_index(name='submission_count')

# Create a new column for formatted date for hover
daily_counts['formatted_date'] = daily_counts['date'].astype(str)

# Plot the number of submissions per day
fig = px.bar(daily_counts, x='date', y='submission_count',
             title="Number of Submissions per Day on r/politics Subreddit (Nov 1–14, 2024)",
             labels={'date': 'Date', 'submission_count': 'Number of Submissions'},
             hover_data={'date': False, 'formatted_date': True, 'submission_count': True})


# Add numbers on top of the bars
fig.update_traces(
    text=daily_counts['submission_count'],  # Add text
    textposition='outside'                 # Position text outside the bar
)

# Customize layout for readability
fig.update_layout(
    xaxis=dict(
        showgrid=True,
        tickformat='%Y-%m-%d',  # Format tick labels to show date (YYYY-MM-DD)
        tickmode='linear',      # Ensure the ticks are correctly spaced
        tickangle=45            # Rotate tick labels for better readability
    ),
    yaxis=dict(showgrid=True),
    title_font_size=16
)

# Save the plot as an image
pio.write_image(fig, "submissions_per_day_politics.png")

# Show the interactive plot
fig.show()
