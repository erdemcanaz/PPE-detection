import os,csv

class dataImporter():  

    def __init__(self, file_path: str) -> None:
        SUPPORTED_EXPORT_TYPES = [".csv"]

        if file_path is None:
            raise ValueError("File path is not provided") 
        self.FILE_PATH = file_path

    def import_csv_as_dict(self) -> list:
        csv_file_path = self.FILE_PATH
        data = []
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data


