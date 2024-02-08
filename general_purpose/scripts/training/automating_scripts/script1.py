import os
 
def process_labels(directory):
    #if the label corresponding to 2 is present, also append the label corresponding to 1 exactly at the same location
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r+') as file:
                lines = file.readlines()
                modified_lines = []
                for line in lines:
                    if line.startswith('2'):
                        new_line = '1' + line[1:]
                        modified_lines.append(new_line)
                if modified_lines:
                    file.writelines(modified_lines)

# Replace 'your_directory_path' with the path to your labels directory.
directory_path = input("Enter the path to your labels directory: ")
process_labels(directory_path)
