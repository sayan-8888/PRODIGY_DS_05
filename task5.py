import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Load the dataset
df = pd.read_csv('NYC.csv')

# Explore the data (optional)
print(df.head())
print(df.info())

# Handle missing values (if necessary)
# For example:
df.dropna(subset=['CRASH DATE', 'CRASH TIME', 'BOROUGH'], inplace=True)

# Extract relevant features
# Assuming 'CRASH DATE' has a format you can convert to datetime
df['DATE'] = pd.to_datetime(df['CRASH DATE'])  # Adjust format if needed

# Extract hour from 'CRASH TIME' (assuming format like HH:MM)
df['Hour'] = df['CRASH TIME'].str.split(':').str[0].astype(int)

# Day of week and month from 'DATE'
df['Day_of_Week'] = df['DATE'].dt.dayofweek
df['Month'] = df['DATE'].dt.month

# Filter out rows with NaN in latitude or longitude
df_filtered = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

# Analyze patterns
#### 5.1. Borough Distribution
sns.countplot(x='BOROUGH', data=df)
plt.title('Distribution of Accidents by Borough')
plt.show()

# You can perform similar countplots for other categorical variables like:
# - NUMBER OF PERSONS INJURED (assuming numerical)
# - NUMBER OF PERSONS KILLED (assuming numerical)
# - ... (other categorical or numerical variables)

#### 5.3. Time of Day
sns.countplot(x='Hour', data=df)
plt.title('Accidents by Hour of the Day')
plt.show()

# Visualize accident hotspots (using filtered data)
map = folium.Map(location=[df_filtered['LATITUDE'].mean(), df_filtered['LONGITUDE'].mean()], zoom_start=12)

for index, row in df_filtered.iterrows():
    folium.Marker(location=[row['LATITUDE'], row['LONGITUDE']]).add_to(map)

map.save('nyc_accident_hotspots.html')

# Additional analysis (correlation, time series, machine learning)