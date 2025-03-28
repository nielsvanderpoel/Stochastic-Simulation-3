import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file with semicolon as delimiter
df = pd.read_csv('C:\\Users\\20223101\\OneDrive - TU Eindhoven\\Desktop\\Stochastic-Simulation-3\\2024-11_rws_filedata.csv', delimiter=";", decimal=",")

df['DatumFileBegin'] = pd.to_datetime(df['DatumFileBegin'], format='%Y-%m-%d').dt.date
df['DatumFileEind'] = pd.to_datetime(df['DatumFileEind'], format='%Y-%m-%d').dt.date
df['TijdFileBegin'] = pd.to_datetime(df['TijdFileBegin'], format='%H:%M:%S').dt.time
df['TijdFileEind'] = pd.to_datetime(df['TijdFileEind'], format='%H:%M:%S').dt.time
df['TotalSeconds_TFB'] = df['TijdFileBegin'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

incidents_per_day = df.groupby('DatumFileBegin').size().reset_index(name='incidents_per_day')
incidents_per_day['day_of_week'] = pd.to_datetime(incidents_per_day['DatumFileBegin']).dt.day_name()
incidents_per_day = incidents_per_day[['DatumFileBegin', 'day_of_week', 'incidents_per_day']] # Reorder columns

# Create custom labels for x-axis
incidents_per_day['custom_label'] = incidents_per_day['DatumFileBegin'].apply(lambda x: str(x.day) + "\n" + incidents_per_day.loc[incidents_per_day['DatumFileBegin'] == x, 'day_of_week'].values[0][0])

# Plot incidents_per_day
# plt.figure(figsize=(12, 6))
# plt.bar(incidents_per_day['custom_label'], incidents_per_day['incidents_per_day'], color='skyblue')
# plt.xlabel('Day of Month')
# plt.ylabel('Number of Incidents')
# plt.title('Incidents Per Day in November 2024')
# plt.xticks(rotation=0, ha='center')  # Keep labels horizontal and centered
# plt.tight_layout()
# plt.show()

# Convert TijdFileBegin to datetime and extract the hour
df['TijdFileBegin'] = pd.to_datetime(df['TijdFileBegin'], format='%H:%M:%S')  # Keep as datetime
df['Hour'] = df['TijdFileBegin'].dt.hour  # Extract hour

# Convert TijdFileEind to datetime and extract the hour
df['TijdFileEind'] = pd.to_datetime(df['TijdFileEind'], format='%H:%M:%S')  # Keep as datetime
df['Hour_End'] = df['TijdFileEind'].dt.hour  # Extract hour

# Group by DatumFileBegin and Hour to calculate incidents per hour for TijdFileBegin
incidents_per_hour_begin = df.groupby(['DatumFileBegin', 'Hour']).size().reset_index(name='incidents_per_hour_begin')

# Group by DatumFileBegin and Hour_End to calculate incidents per hour for TijdFileEind
incidents_per_hour_end = df.groupby(['DatumFileBegin', 'Hour_End']).size().reset_index(name='incidents_per_hour_end')

# Sum up all incidents for every hour across all days for TijdFileBegin
total_incidents_per_hour_begin = incidents_per_hour_begin.groupby('Hour')['incidents_per_hour_begin'].sum().reset_index()

# Sum up all incidents for every hour across all days for TijdFileEind
total_incidents_per_hour_end = incidents_per_hour_end.groupby('Hour_End')['incidents_per_hour_end'].sum().reset_index()

# Rename columns for clarity
total_incidents_per_hour_begin.columns = ['Hour', 'total_incidents_begin']
total_incidents_per_hour_end.columns = ['Hour', 'total_incidents_end']

# Merge the two datasets on the Hour column
total_incidents_combined = pd.merge(total_incidents_per_hour_begin, total_incidents_per_hour_end, on='Hour', how='outer').fillna(0)

# Plot total incidents per hour for both TijdFileBegin and TijdFileEind
plt.figure(figsize=(12, 6))
plt.bar(total_incidents_combined['Hour'] - 0.2, total_incidents_combined['total_incidents_begin'], width=0.4, label='TijdFileBegin', color='skyblue')
plt.bar(total_incidents_combined['Hour'] + 0.2, total_incidents_combined['total_incidents_end'], width=0.4, label='TijdFileEind', color='orange')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Number of Incidents')
plt.title('Total Incidents Per Hour Across All Days (Begin vs End)')
plt.xticks(range(0, 24))  # Ensure all hours (0-23) are shown on the x-axis
plt.legend()  # Add a legend to distinguish the two datasets
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability
plt.tight_layout()
plt.show()
