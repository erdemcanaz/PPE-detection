import scripts.data_importer as data_importer
import pprint

#========================DEFINE DATA EXPORTER-IMPORTER OBJECTS=================================
csv_file_path = input("Enter the csv file path to import the results: ")
csv_importer = data_importer.dataImporter(file_path= csv_file_path)


detection_data = csv_importer.import_csv_as_dict()

import numpy as np
import matplotlib.pyplot as plt

x_vals = []
y_vals = []


detection_seconds = {}
for detection_dict in detection_data:
    video_time = detection_dict["video_time"].split(":")
    video_seconds = int(video_time[0]) * 3600 + int(video_time[1]) * 60 + int(video_time[2])
    x_val = detection_dict["person_x"]
    y_val = detection_dict["person_y"]
    x_vals.append(x_val)
    y_vals.append(y_val)

    if video_seconds not in detection_seconds:
        detection_seconds[video_seconds] = 1
    else:
        detection_seconds[video_seconds] += 1
    


#========================PLOT THE DETECTIONS=================================
#time vs detection count
        
import matplotlib.pyplot as plt
import numpy as np

# Constants for the time frame and interval duration
total_seconds = 20037
interval_seconds = 5 * 60  # 5 minutes in seconds
num_intervals = total_seconds // interval_seconds + (1 if total_seconds % interval_seconds else 0)

# Initialize a list to hold the maximum number of people detected per interval
interval_max_people = [0] * num_intervals

# Populate the list with the maximum number of people detected in each interval
for second, count in detection_seconds.items():
    interval_index = second // interval_seconds
    if interval_index < num_intervals:
        interval_max_people[interval_index] = max(interval_max_people[interval_index], count)

# Generate x-axis labels for the intervals, displaying only selected labels to avoid overcrowding
x_labels = [f'{i*5} dk' if i % 6 == 0 else '' for i in range(num_intervals)]

# Specify the RGB color as a tuple
rgb_color = (0.49, 0.807, 0.929)  # A shade of teal

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(range(num_intervals), interval_max_people, color=rgb_color)


plt.xlabel('Time Interval')
plt.ylabel('Tespit edilen maksimum kişi sayısı')
plt.title('Max People Detected in 5-Min Intervals Over 20037 Seconds')
plt.xticks(range(num_intervals), x_labels, rotation=45)

plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()


#DENSITY PLOT
# Creating a 25x25 grid
grid_size = 35
hist, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=grid_size, range=[[0, 10], [0, 10]])

# Plotting the intensity plot
plt.figure(figsize=(8, 6))
plt.imshow(hist.T, origin='lower', extent=[0, 10, 0, 10], aspect='auto', cmap='Blues')
plt.colorbar(label='Hücre başına tespit sayısı')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Intensity Plot of Detections')
plt.show()
