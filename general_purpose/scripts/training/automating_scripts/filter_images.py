import cv2
import os
import shutil

# Directories
image_dir = input("Enter the path to your images folder: ")
label_dir = input("Enter the path to your labels folder: ")
filtered_image_dir = input("Enter the path to your filtered images folder: ")
filtered_label_dir = input("Enter the path to your filtered labels folder: ")

assert image_dir != filtered_image_dir, "Image directory and filtered image directory cannot be the same."
assert label_dir != filtered_label_dir, "Label directory and filtered label directory cannot be the same."


# Create filtered directories if they don't exist
os.makedirs(filtered_image_dir, exist_ok=True)
os.makedirs(filtered_label_dir, exist_ok=True)

# Get the list of image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
current_image_index = 0

while True:
    # Load current image
    image_path = os.path.join(image_dir, image_files[current_image_index])
    image = cv2.imread(image_path)
    cv2.imshow('Image', image)

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    # 's' key - copy and move to next
    if key == ord('s'):
        # Check if corresponding label file exists
        label_file_name = image_files[current_image_index].split('.')[0] + '.txt'
        label_path = os.path.join(label_dir, label_file_name)

        if os.path.exists(label_path):
            # Copy the label file
            shutil.copy2(label_path, os.path.join(filtered_label_dir, label_file_name))
            print(f"Copied label for {image_files[current_image_index]}")

            # Copy the image file
            shutil.copy2(image_path, os.path.join(filtered_image_dir, image_files[current_image_index]))
            print(f"Copied {image_files[current_image_index]}")
        else:
            print(f"No label file found for {image_files[current_image_index]}. Image not copied.")
        # 'd' key - move to next image
    if key == ord('d'):
        current_image_index = (current_image_index + 1) % len(image_files)

    # 'a' key - move to previous image
    elif key == ord('a'):
        current_image_index = (current_image_index - 1) % len(image_files)

    # 'q' key - quit
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
