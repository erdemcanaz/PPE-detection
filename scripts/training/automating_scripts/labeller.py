from ultralytics import YOLO
import cv2
import math, os

model_path = input("Enter the path to your model: ")
images_path = input("Enter the path to your images: ")
labels_path = input("Enter the path to your labels: ")

yolo_object = YOLO(model_path)
                  
def detect_and_update_frame(frame, img_name_wo_extension = "", label_folder_path = "", confidence_threshold = 0.2):
    global yolo_object
    classNames = ['yuz', 'yok-baret', 'insan','kafa', 'var-baret']
    
    results = yolo_object(frame, stream=True)

    coordinates = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_no = int(box.cls[0])   
            class_name = classNames[class_no]

            if class_name not in ['insan']:
                continue

            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            print("frame size", frame_width, frame_height)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)          
            conf = math.ceil((box.conf[0] * 100)) / 100
         
            if(conf < confidence_threshold):
                continue

            label = f'{class_name} : {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
            cv2.rectangle(frame, (x1, y1), c2, (0,255,0), -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

            print("box data", x1, y1, x2, y2, frame_width, frame_height)
            x1 = x1/frame_width
            y1 = y1/frame_height
            x2 = x2/frame_width
            y2 = y2/frame_height
            coordinates = [ x1, y1, x2, y2]
            print("box data",coordinates, frame_width, frame_height)
            new_line = f"0 {x1} {y1} {x2} {y2}\n"

            # Append the line to a text file
            output_file = os.path.join(label_folder_path, f"{img_name_wo_extension}.txt")
            with open(output_file, "a") as file:
                file.write(new_line)

    return frame 

def resize_image(image, window_width, window_height):
    # Calculate the aspect ratio
    height, width = image.shape[:2]
    img_aspect = width / height

    # Calculate the scaling factors
    if width > window_width or height > window_height:
        if img_aspect > 1:
            # The image is wide
            scale = window_width / width
        else:
            # The image is tall
            scale = window_height / height
    else:
        # No scaling required
        scale = 1

    # Resize the image
    return cv2.resize(image, (int(width * scale), int(height * scale)))

def main_human_inspection(folder_path):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the folder")
        return

    current_index = 0

    while True:
        # Load image
        img_path = os.path.join(folder_path, image_files[current_index])
        img = cv2.imread(img_path)

        # Modify image
        modified_img = detect_and_update_frame(img)

        # Display image
        resize_image(modified_img, 1920, 1080)
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

def main_labell_all(image_folder_path, label_folder_path):
    # List all files in the folder
    files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the folder")
        return

    current_index = 0

    while True:
        # Load image
        img_path = os.path.join(image_folder_path, image_files[current_index])
        img_name = os.path.basename(img_path).rsplit('.', 1)[0]
        img = cv2.imread(img_path)

        # Modify image
        modified_img = detect_and_update_frame(img,  img_name_wo_extension = img_name, label_folder_path = label_folder_path)

        # Display image
        resize_image(modified_img, 1920, 1080)
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

if __name__ == "__main__":
    main_labell_all(images_path, labels_path)
        