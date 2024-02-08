import cv2
import os
import glob

# Path to the directory containing images and labels
image_folder = input("Enter the path to your image folder: ")
label_folder = input("Enter the path to your label folder: ")

# Getting list of image paths
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

def delete_image_and_label(image_path):
    label_path = os.path.join(label_folder, os.path.basename(image_path).replace('.jpg', '.txt'))
    os.remove(image_path)
    if os.path.exists(label_path):
        os.remove(label_path)

current_index = 0

while True:
    if current_index < 0:
        current_index = 0
    elif current_index >= len(image_paths):
        current_index = len(image_paths) - 1

    if len(image_paths) == 0:
        print("No images left in the folder.")
        break

    # Load and display image
    image_path = image_paths[current_index]
    image = cv2.imread(image_path)
    
    # Resize the image while maintaining aspect ratio
    width = 800  # Specify the desired width
    height = int(image.shape[0] * width / image.shape[1])  # Calculate the corresponding height
    resized_image = cv2.resize(image, (width, height))
    cv2.imshow('Image Viewer', resized_image)

    # Wait for the user to press a key
    key = cv2.waitKey(0) & 0xFF

    # Check which key is pressed
    if key == ord('d'):  # Next image
        current_index += 1
    elif key == ord('a'):  # Previous image
        current_index -= 1
    elif key == ord('w'):  # Delete image and label
        delete_image_and_label(image_path)
        del image_paths[current_index]
        print(f"Deleted {image_path}")
        continue
    elif key == ord('q'):  # Quit program
        break

cv2.destroyAllWindows()
