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
likelihoods = {}
for person_id in tracked_persons:
    likelihood = data_analyzer.calculate_direct_passage_likelyhood(person_dict=tracked_persons[person_id], min_data_points=5, min_confidence=0.5, x_border=8.25, x_border_width=1.75)
    likelihoods[person_id] = likelihood

# Sort the person_ids by their likelihood in descending order
sorted_person_ids = sorted(likelihoods, key=likelihoods.get, reverse=True)

print(sorted_person_ids)

# Print the sorted person_ids
for person_id in sorted_person_ids:
    time_stamp = data_analyzer.get_timestamp_of_a_record(tracked_persons[person_id])
    print(f"{time_stamp} | Person ID: {person_id}  Likelihood: {likelihoods[person_id]}")

