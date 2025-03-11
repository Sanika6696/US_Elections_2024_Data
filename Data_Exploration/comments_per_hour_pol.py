import pandas as pd
import json
import plotly.express as px
from datetime import datetime
import plotly.io as pio

# Read the data
data = pd.read_csv("posts_202412021541.csv", on_bad_lines='skip')

# Extract the 'now' field from the JSON in the 'data' column
def extract_now(json_string):
    try:
        parsed_data = json.loads(json_string)
        return parsed_data.get("now", None)  # Extract the 'now' field
    except:
        return None

# Apply the extraction function to the 'data' column
data['now'] = data['data'].apply(extract_now)

# Drop rows where the 'now' field is missing
data = data.dropna(subset=['now'])

# Convert 'now' to datetime format for filtering
data['datetime'] = pd.to_datetime(data['now'], format='%m/%d/%y(%a)%H:%M:%S', errors='coerce')

# Filter for the '/pol/' board
data_pol = data[data['board'] == 'pol']

# Define the date range for filtering with time information
start_date = '2024-11-01 00:00:00'  # Specify the time as well
end_date = '2024-11-14 23:59:59'    # Specify the time as well

# Convert the date strings to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter for the date range
data_pol = data_pol[(data_pol['datetime'] >= start_date) & (data_pol['datetime'] <= end_date)]

# Group by hour and count the number of comments
data_pol['hour'] = data_pol['datetime'].dt.floor('h')  # Round down to nearest hour
hourly_counts = data_pol.groupby('hour').size().reset_index(name='count')
hourly_counts['hover_data'] = hourly_counts['hour'].dt.strftime('%Y-%m-%d %H:%M')

# Interactive plot using Plotly
fig = px.line(hourly_counts, x='hour', y='count', title="Number of Comments per Hour on 4chan's /pol/ (Nov 1–14, 2024)",
              labels={'hour': 'Date wise comments binned hourly', 'count': 'Number of Comments'}, markers=True)

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
pio.write_image(fig, "comments_per_hour_pol.png")

# Show the interactive plot
fig.show()
