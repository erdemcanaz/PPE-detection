from ultralytics import YOLO
import torch
import cv2,math,time,os
import sys
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#==============================
model_path = "C:\\Users\\Levovo20x\\Documents\\GitHub\\PPE-detection\\general_purpose\\scripts\\object_detection\\models\\TETT_15_02_2024_v2.pt"
#==============================

yolo_object_TETT = YOLO(model_path)
print(yolo_object_TETT.names)

def detect_and_update_frame(frame, conf_human = 0.2):
    global yolo_object_TETT
    classNames = ['tett_region', 'barcode_region', 'qr_region']
    results = yolo_object_TETT(frame, stream=True, verbose = False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_no = int(box.cls[0])   
            class_name = classNames[class_no]

            # if(class_name not in ['yok-baret', 'var-baret']):
            #     continue
        
            if class_name not in ['tett_region']:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
             
            color = (0,255,0)

            label = f'{class_name} : {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3          

            # Crop the region of interest from the original frame
            TETT_region_frame = frame[(y1+15):(y2-15), (x1):(x2)]
            
            # Define your scaling factors
            frame_scale = 4  #       

            # Calculate the new size
            new_width = int(TETT_region_frame.shape[1] * frame_scale)
            new_height = int(TETT_region_frame.shape[0] * frame_scale)
            new_size = (new_width, new_height)

            # Resize the image
            enlarged_image = cv2.resize(TETT_region_frame, new_size, interpolation=cv2.INTER_CUBIC)

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(enlarged_image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to binarize the image
            # You might need to adjust the method and parameters depending on your image
            _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply dilation and erosion to remove some noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

            # Run Tesseract OCR on the preprocessed image
            text = pytesseract.image_to_string(processed_image, lang='eng')
            print("text:",text)

            #print(conf, class_name, enlarged_image.shape)
            #Display the cropped region in a new window (or you can process/save as needed)
            cv2.imshow("Cropped Region", binary_image)
            cv2.waitKey(1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)


            
    return frame 


if __name__ == "__main__":
    # Open the video file
    cap = cv2.VideoCapture(1,  cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't open the video file.")
            break

        frame = detect_and_update_frame(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()