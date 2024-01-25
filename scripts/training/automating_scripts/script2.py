import os
import shutil

def copy_labeled_images(image_dir, label_dir, labeled_image_dir, labeled_label_dir):
    # Create directories if they don't exist
    os.makedirs(labeled_image_dir, exist_ok=True)
    os.makedirs(labeled_label_dir, exist_ok=True)

    counter = 1
    for image_file in os.listdir(image_dir):
        if image_file.endswith((".png", ".jpg", ".jpeg")):  # Add or modify extensions as needed
            label_file = image_file.rsplit('.', 1)[0] + '.txt'
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)

            if os.path.exists(label_path):
                # Copy image
                shutil.copy(image_path, os.path.join(labeled_image_dir, image_file))
                # Copy label
                shutil.copy(label_path, os.path.join(labeled_label_dir, label_file))
                print(f"{counter}:Successfully copied {image_file} and {label_file}")
                counter += 1

# Define your directories here
image_directory = input("Enter the path to your images directory: ")
label_directory = input("Enter the path to your labels directory: ")
labeled_image_directory = input("Enter where the labeled images will be copied to")
labeled_label_directory = input("Enter where the labels will be copied to")

copy_labeled_images(image_directory, label_directory, labeled_image_directory, labeled_label_directory)
