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


tracked_persons = {}

#### Tracking dict format: ##########
#   'box_confidence': '0.8955957',
#   'id': '217',
#   'person_x': '4.775367312138306',
#   'person_y': '2.728398447112553',
#   'person_z': '1.190439820016809',
#   'px': '662.0',
#   'py': '429.5',
#   'state': '1',
#   'timestamp': '19938.941176470587'}

for tracking_dict in tracking_data:
    person_id = tracking_dict["id"]
    if person_id not in tracked_persons:
        tracked_persons[person_id] = []
    tracked_persons[person_id].append(tracking_dict)

k = 0
for person_id in tracked_persons:
    r = data_analyzer.is_direct_passage(tracked_persons[person_id], min_data_points= 5, min_confidence = 0.45,  x_min_threshold=7.75, x_max_threshold=8.75)
    if r:
        k+=1
print(f"Total number of people who passed through the restricted area: {k}")