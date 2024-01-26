from ultralytics import YOLO
import torch
import cv2,math,time,os
import sys

#==============================
model_path = "scripts/object_detection/models/ppe_26_01_2024.pt"
#==============================

yolo_object = YOLO(model_path)
print(yolo_object.names)

                  
def detect_and_update_frame(frame, confidence_threshold = 0.2):
    global yolo_object
    {0: 'yuz', 1: 'yok-baret', 2: 'insan', 3: 'kafa', 4: 'var-baret'}
    classNames = ['yuz', 'yok-baret', 'insan','kafa', 'var-baret']
    
    results = yolo_object(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_no = int(box.cls[0])   
            class_name = classNames[class_no]

            # if(class_name not in ['yok-baret', 'var-baret']):
            #     continue
        
            if class_name not in ['insan']:
                continue

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

    return frame 

