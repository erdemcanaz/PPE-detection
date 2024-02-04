import pprint
import scripts.data_importer as data_importer
import scripts.data_analyzer as data_analyzer

import os
if os.name == 'nt':
    os.system('cls')
    # For Unix/Linux or macOS, use 'clear'
else:
    os.system('clear')

#========================DEFINE DATA EXPORTER-IMPORTER OBJECTS=================================
csv_file_path = input("Enter the csv file path to import the results: ")
csv_importer = data_importer.dataImporter(file_path= csv_file_path)

tracking_data = csv_importer.import_csv_as_dict()


tracked_persons = {} # a dict that the tracking id as key and all the related tracking data as value
for tracking_dict in tracking_data:
    person_id = tracking_dict["id"]
    if person_id not in tracked_persons:
        tracked_persons[person_id] = []
    tracked_persons[person_id].append(tracking_dict)

# Calculate likelihood for each person_id
all_likelyhoods = []
for person_id in tracked_persons:
    result_dict = data_analyzer.calculate_direct_passage_likelyhood(person_dict=tracked_persons[person_id], min_data_points=5, min_confidence=0.5, x_border=7.5, x_border_width=5)
    all_likelyhoods.append(result_dict)
#sort the person_dict by timestamp
    

all_likelyhoods_sorted = sorted(all_likelyhoods, key=lambda x: x['likelihood'], reverse=True)

# Print the sorted person_ids
for tracking_data in all_likelyhoods_sorted:
    start_time = tracking_data["start_time"]
    start_seconds = tracking_data["start_seconds"]

    end_time = tracking_data["end_time"]
    end_seconds = tracking_data["end_seconds"]

    time_stamp = f"{start_time} - {end_time}"
    likelihood = tracking_data["likelihood"]

    window_size = max(5, len(tracking_data["x_vals"])//10)
    x_vals = tracking_data["x_vals"]
    x_vals_mean_filter = data_analyzer.apply_mean_filter(x_vals, window_size=5)

    y_vals = tracking_data["y_vals"]
    y_vals_mean_filter = data_analyzer.apply_mean_filter(y_vals, window_size=5)

    print(f"{time_stamp} | Tracking ID: {person_id}  Likelihood: {likelihood}")

    title = f"Tracking ID: {person_id} | Likelihood: {likelihood:.2f} | {time_stamp}"
    data_analyzer.plot_xy_tracking_results(x_vals = x_vals_mean_filter, y_vals = y_vals_mean_filter, title=title)
