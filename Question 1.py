import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import kstest

# Read CSV file with semicolon as delimiter
df = pd.read_csv('C:\\Users\\20223101\\OneDrive - TU Eindhoven\\Desktop\\Stochastic-Simulation-3\\2024-11_rws_filedata.csv', delimiter=";", decimal=",")

df['DatumFileBegin'] = pd.to_datetime(df['DatumFileBegin'], format='%Y-%m-%d').dt.date
df['DatumFileEind'] = pd.to_datetime(df['DatumFileEind'], format='%Y-%m-%d').dt.date
df['TijdFileBegin'] = pd.to_datetime(df['TijdFileBegin'], format='%H:%M:%S').dt.time
df['TijdFileEind'] = pd.to_datetime(df['TijdFileEind'], format='%H:%M:%S').dt.time
df['TotalSeconds_TFB'] = df['TijdFileBegin'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

# Ensure the DataFrame is sorted by DatumFileBegin and TijdFileBegin
df = df.sort_values(by=['DatumFileBegin', 'TijdFileBegin'])

# Combine DatumFileBegin and TijdFileBegin into a single datetime column for accurate time calculations
df['DatetimeFileBegin'] = pd.to_datetime(df['DatumFileBegin'].astype(str) + ' ' + df['TijdFileBegin'].astype(str))

df['FileDuurInSeconds'] = df['FileDuur'] * 60

# Calculate the time difference between successive incidents
df['TimeElapsed'] = df['DatetimeFileBegin'].diff().dt.total_seconds()

# Reset the time difference to NaN for the first incident of each day
df.loc[df['DatumFileBegin'] != df['DatumFileBegin'].shift(), 'TimeElapsed'] = None


incidents_per_day = df.groupby('DatumFileBegin').size().reset_index(name='incidents_per_day')
incidents_per_day['day_of_week'] = pd.to_datetime(incidents_per_day['DatumFileBegin']).dt.day_name()
incidents_per_day = incidents_per_day[['DatumFileBegin', 'day_of_week', 'incidents_per_day']] # Reorder columns

# Create custom labels for x-axis
incidents_per_day['custom_label'] = incidents_per_day['DatumFileBegin'].apply(lambda x: str(x.day) + "\n" + incidents_per_day.loc[incidents_per_day['DatumFileBegin'] == x, 'day_of_week'].values[0][0])

# Plot incidents_per_day
plt.figure(figsize=(12, 6))
plt.bar(incidents_per_day['custom_label'], incidents_per_day['incidents_per_day'], color='skyblue')
plt.xlabel('Day of Month')
plt.ylabel('Number of Incidents')
plt.title('Incidents Per Day in November 2024')
plt.xticks(rotation=0, ha='center')  # Keep labels horizontal and centered
plt.tight_layout()
plt.show()

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

# Group by Hour and calculate the total number of incidents per hour
incident_rate_per_hour = df.groupby('Hour').size().reset_index(name='total_incidents')

# Calculate the average incident rate per hour
incident_rate_per_hour['incident_rate'] = incident_rate_per_hour['total_incidents'] / len(df['DatumFileBegin'].unique())

print(incident_rate_per_hour)

# Ensure FileDuur is numeric and drop NaN values
df['FileDuur'] = pd.to_numeric(df['FileDuur'], errors='coerce')  # Convert to numeric if not already
file_duur_data = df['FileDuur'].dropna()


# Fit an exponential distribution to the data
params = stats.expon.fit(file_duur_data)  # Fit the exponential distribution
loc, scale = params  # Extract location and scale parameters

# Generate x values for the fitted distribution
x = np.linspace(file_duur_data.min(), file_duur_data.max(), 100)
pdf = stats.expon.pdf(x, loc=loc, scale=scale)  # Probability density function of the fitted distribution

# Plot the histogram of the data
plt.figure(figsize=(10, 6))
plt.hist(file_duur_data, bins=50, density=True, alpha=0.6, color='skyblue', label='FileDuur Data')  # Increased bins to 50

# Plot the fitted distribution
plt.plot(x, pdf, 'r-', label=f'Fitted Exponential (loc={loc:.2f}, scale={scale:.2f})')

# Add labels and title
plt.xlabel('FileDuur')
plt.ylabel('Density')
plt.title('Histogram of Duration of an Incident with Fitted Exponential Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print the fitted parameters
print(f"Fitted Exponential Distribution Parameters: loc={loc:.2f}, scale={scale:.2f}")

# Perform the Kolmogorov-Smirnov test
ks_statistic, p_value = kstest(file_duur_data, 'expon', args=(loc, scale))

# Print the results of the KS test
print(f"Kolmogorov-Smirnov Test Statistic: {ks_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpret the results
alpha = 0.05  # Significance level
if p_value > alpha:
    print("Fail to reject the null hypothesis: The data follows the fitted exponential distribution.")
else:
    print("Reject the null hypothesis: The data does not follow the fitted exponential distribution.")

# Fit a gamma distribution to the data
params_gamma = stats.gamma.fit(file_duur_data)
shape, loc, scale = params_gamma

# Generate x values for the fitted distribution
pdf_gamma = stats.gamma.pdf(x, 1.19, loc, 6.09)

# Plot the histogram and fitted gamma distribution
plt.figure(figsize=(10, 6))
plt.hist(file_duur_data, bins=50, density=True, alpha=0.6, color='skyblue', label='FileDuur Data')
plt.plot(x, pdf_gamma, 'm-', label=f'Fitted Gamma (shape={shape:.2f}, scale={scale:.2f})')
plt.xlabel('FileDuur')
plt.ylabel('Density')
plt.title('Histogram of Duration of an Incident with Fitted Gamma Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Perform the Kolmogorov-Smirnov test for gamma
ks_statistic_gamma, p_value_gamma = kstest(file_duur_data, 'gamma', args=(shape, loc, scale))
print(f"Gamma KS Test Statistic: {ks_statistic_gamma:.4f}")
print(f"Gamma P-Value: {p_value_gamma:.4f}")

# Fit a log-normal distribution to the data
params_lognorm = stats.lognorm.fit(file_duur_data, floc=0)  # Fix location to 0 for log-normal
shape, loc, scale = params_lognorm

# Generate x values for the fitted distribution
pdf_lognorm = stats.lognorm.pdf(x, shape, loc, scale)

# Plot the histogram and fitted log-normal distribution
plt.figure(figsize=(10, 6))
plt.hist(file_duur_data, bins=50, density=True, alpha=0.6, color='skyblue', label='Duration of an Incident Data')
plt.plot(x, pdf_lognorm, 'g-', label=f'Fitted Log-Normal (shape={shape:.2f}, scale={scale:.2f})')
plt.xlabel('FileDuur')
plt.ylabel('Density')
plt.title('Histogram of Duration of an Incident with Fitted Log-Normal Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Perform the Kolmogorov-Smirnov test for log-normal
ks_statistic_lognorm, p_value_lognorm = kstest(file_duur_data, 'lognorm', args=(shape, loc, scale))
print(f"Log-Normal KS Test Statistic: {ks_statistic_lognorm:.4f}")
print(f"Log-Normal P-Value: {p_value_lognorm:.4f}")