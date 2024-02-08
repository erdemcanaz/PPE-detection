import os,csv

class CSV_Exporter():  

    def __init__(self, folder_path: str, file_name_wo_extension:str) -> None:
        if folder_path is None:
            raise ValueError("Folder path is not provided")
        if not os.path.isdir(folder_path):
            raise ValueError("Given folder path does not exist")
        if file_name_wo_extension is None:
            raise ValueError("File name without extension is not provided")
        
        self.FILE_NAME_WO_EXTENSION = file_name_wo_extension        
        self.FILE_PATH = os.path.join(folder_path, file_name_wo_extension + ".csv")

    def append_row(self, row_dict_to_append:dict) -> None:
        # The CSV file path
        csv_file_path = self.FILE_PATH

        # Check if the CSV file already exists
        file_exists = os.path.isfile(csv_file_path)

        # Open the file in append mode, which will create it if it doesn't exist
        with open(csv_file_path, mode='a', newline='') as file:
            # Define the fieldnames based on the dictionary keys
            fieldnames = row_dict_to_append.keys()
            
            # Create a DictWriter object
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # If the CSV file didn't exist, write the header
            if not file_exists:
                writer.writeheader()
            
            # Append the dictionary as a row in the CSV
            writer.writerow(row_dict_to_append)


