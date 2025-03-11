import pandas as pd
import plotly.express as px
import plotly.io as pio

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('subreddit_posts_comment_202412021708.csv')

# Convert 'comment_date' to datetime format (automatically inferred)
df['comment_date'] = pd.to_datetime(df['comment_date'])

# Filter the data to only include the r/politics subreddit
df_politics = df[df['subreddit'] == 'politics']

# Define the date range for filtering with time information
start_date = '2024-11-01 00:00:00'  # Specify the time as well
end_date = '2024-11-14 23:59:59'    # Specify the time as well

# Convert the date strings to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter for the date range
df_politics = df_politics[(df_politics['comment_date'] >= start_date) & (df_politics['comment_date'] <= end_date)]

# Group by hour and count the number of comments
df_politics['hour'] = df_politics['comment_date'].dt.floor('h')  # Round down to nearest hour
hourly_counts = df_politics.groupby('hour').size().reset_index(name='count')

# Create a new column for formatted date/time for hover
hourly_counts['formatted_hour'] = hourly_counts['hour'].dt.strftime('%Y-%m-%d %H:%M:%S')

# # Interactive plot using Plotly
fig = px.area(hourly_counts, x='hour', y='count', title="Number of Comments per Hour on r/politics Subreddit (Nov 1–14, 2024)",
              labels={'hour': 'Date wise comments binned hourly', 'count': 'Number of Comments'}, 
              hover_data={'hour': False, 'formatted_hour': True, 'count': True},
              markers=True)

# Customize layout for readability
fig.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=False),  # Enable scrollable range slider
        showgrid=True,
        tickformat='%b %d',             # Format tick labels to show date (e.g., Nov 01, Nov 02)
        tickmode='linear',               # Make sure the ticks are correctly spaced
        dtick="86400000",               # Ensures the tick interval is one day (24 hours)
        tickangle=45                    # Rotate tick labels for better readability
    ),
    yaxis=dict(showgrid=True),
    title_font_size=16
)

# Save the plot as an image
pio.write_image(fig, "comments_per_hr_r_politics.png")

# Show the interactive plot
fig.show()
