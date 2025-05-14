# Extract HRV features from 24h long ECG data and create a json file (used for analysis, and Chart.js real-time visualization
import astropy
import future
import nolds
import numpy
import pandas as pd
import numpy as np
import datetime
import hrvanalysis
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features, get_frequency_domain_features, get_time_domain_features, get_geometrical_features, get_csi_cvi_features, get_poincare_plot_features
from hrvanalysis.plot import VlfBand, LfBand, HfBand

print(dir(hrvanalysis))



# Read the CSV file into a DataFrame
df = pd.read_csv('signal_data.csv')

# Store each column as a list
timestamp = df['Timestamp'].tolist()
rr = df['RR-intervals'].tolist()
print("first", rr[:5])
rr = [float(i) for i in rr]
print("first", rr[:5])
raw_rr_data_combined = np.array(rr)
# This remove outliers from signal
rr_intervals_without_outliers = remove_outliers(rr_intervals=raw_rr_data_combined,low_rri=300, high_rri=2000)
# This replaces outliers nan values with linear interpolation
interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,interpolation_method="linear")
# This remove ectopic beats from signal
nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")

# This replaces ectopic beats nan values with linear interpolation
interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
rr_data_cleaned = interpolated_nn_intervals

# Calculate the average RR interval time
mean_rr_time = np.mean(rr_data_cleaned)
print("Average interval time: ", mean_rr_time)

# Estimate the sample rate
sample_rate_estimate = 1 / (mean_rr_time/1000)

print("Estimated sample rate:", sample_rate_estimate, "Hz")

# Assume a starting time
start_time = datetime.datetime(2024, 2, 7, 6, 54, 32)  # Example starting time

# Extract last millisecond from the starting time
last_millisecond = start_time.microsecond

# Create a copy of the cleaned RR data
cleaned_rr_data_modified = rr_data_cleaned.copy()

# Adjust the first RR interval by adding the last millisecond from the starting time
cleaned_rr_data_modified[0] += last_millisecond

# Calculate cumulative sum of RR-intervals
cumulative_sum = np.cumsum(cleaned_rr_data_modified)

# Create timestamps
timestamps = [start_time + datetime.timedelta(milliseconds=ms) for ms in cumulative_sum]

# Create DataFrame directly from timestamps
df_time = pd.DataFrame({"Timestamp": timestamps})

# Convert timestamps to Unix timestamps (numerical values)
timestamps_unix = [ts.timestamp() for ts in timestamps]

# Create DataFrame with timestamps and signal values
df = pd.DataFrame({'Timestamp': timestamps_unix, 'RR-intervals': rr_data_cleaned})

def sliding_window_analysis(signal, timestamps, window_length, overlap_percentage):
    results = []
    overlap = int(window_length * (overlap_percentage / 100))
     # Assuming 'v' is a list of NN intervals
    vlf_band = VlfBand(low=0.003, high=0.04)
    lf_band = LfBand(low=0.04, high=0.15)
    hf_band = HfBand(low=0.15, high=0.4)
    for i in range(0, len(signal) - window_length + 1, overlap):
        window = signal[i:i + window_length]
        window_timestamps_ = timestamps[i:i + window_length]
        window_mean = np.mean(window)
        window_center_time = np.mean(window_timestamps_)
        # window_features = get_frequency_domain_features(window)
        window_features = get_frequency_domain_features(window, vlf_band=vlf_band, lf_band=lf_band, hf_band=hf_band)
        time_d_features = get_time_domain_features(window)
        nonlin_features = get_csi_cvi_features(window)
        nonlin_poincare_features = get_poincare_plot_features(window)
        results.append((window_center_time, window_mean, window_features, time_d_features, nonlin_features, nonlin_poincare_features))
    return results

# Set the window length and overlap percentage
window_length = 100  # Replace with your desired window length (number of data points)
overlap_percentage = 90  # Replace with your desired overlap percentage

# Perform sliding window analysis
window_results = sliding_window_analysis(df['RR-intervals'].values, df['Timestamp'].values, window_length, overlap_percentage)

# Unzip the results to get timestamps and filtered signal
window_center_time, full_signal_with_time, freq_d_features, time_d_features, nonlin_features, poincare_features = zip(*window_results)

# Extract all frequency-domain features

linear_feature_names = list(time_d_features[0].keys())
freq_feature_names = list(freq_d_features[0].keys())
nonlin_feature_names = list(nonlin_features[0].keys())
poincare_feature_names = list(poincare_features[0].keys())

data = {"window_center_time": window_center_time, "full_signal_with_time": full_signal_with_time,
        "freq_d_features": freq_d_features, "time_d_features": time_d_features,
        "poincare_features": poincare_features,
        "nonlin_features": nonlin_features}



import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


#with open('hrv_data_5.json', 'w') as json_file:
 #   json.dump(data, json_file)
transformed_data = []
for elem in window_results:
    transformed_data.append({
    'window_start': elem[0],
    'mean_rr': elem[1],
    'frequency': elem[2],
    'time_data': elem[3],
    'nonlinear': elem[4],
    'poincare': elem[5]})

json_string = json.dumps(transformed_data, cls=NpEncoder)

# Now you can write the JSON string to a file if needed
with open('feature_data.json', 'w') as json_file:
    json_file.write(json_string)
