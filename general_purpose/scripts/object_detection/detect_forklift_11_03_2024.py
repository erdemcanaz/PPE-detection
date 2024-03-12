from ultralytics import YOLO
import torch
import cv2,math,time,os
import sys

#==============================
model_path = "C:/Users/Levovo20x/Documents/GitHub/PPE-detection/scripts/object_detection/models/secret_forklift_11_03_2024.pt"
#==============================

yolo_object_forklift = YOLO(model_path)
print(yolo_object_forklift.names)

def detect_and_update_frame(frame, conf_forklift = 0.2):
    global yolo_object_forklift
    classNames = ['forklift']
    results = yolo_object_forklift(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_no = int(box.cls[0])   
            class_name = classNames[class_no]

            # if(class_name not in ['yok-baret', 'var-baret']):
            #     continue
        
            if class_name not in ['forklift']:
                continue


            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
         
            if conf < conf_forklift:
                continue

            color = (0,255,0)

            label = f'{class_name} : {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)

            cv2.putText(frame, label, (x1, y1), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

    return frame 

