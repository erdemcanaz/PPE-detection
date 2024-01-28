import os
import cv2,math,time,os

import scripts.object_detection.detect_ppe_forklift_28_01_2024 as ppe_forklift_detector

folder_path = input("Enter the path to your images: ")

# List all files in the folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print("No images found in the folder")
    raise Exception("No images found in the folder")

current_index = 0

while True:
    # Load image
    img_path = os.path.join(folder_path, image_files[current_index])
    img = cv2.imread(img_path)

    # Modify image
    modified_img = ppe_forklift_detector.detect_and_update_frame(img)

    # Display image
    cv2.imshow('Image Viewer', modified_img)

    # Wait for key press
    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):
        # Next image
        current_index = (current_index + 1) % len(image_files)
    elif key == ord('a'):
        # Previous image
        current_index = (current_index - 1) % len(image_files)
    elif key == ord('q'):
        # Quit the program
        break

cv2.destroyAllWindows()