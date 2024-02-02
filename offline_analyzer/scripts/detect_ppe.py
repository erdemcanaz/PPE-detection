from ultralytics import YOLO
import torch
import cv2,math,time,os
import sys


class ppeDetector():

    def __init__(self, model_path): 

        if os.path.exists(model_path):
        # Check if the file has the specified extension
            _, file_extension = os.path.splitext(model_path)
            if file_extension != ".pt":
                raise ValueError(f"The model file should have the extension '.pt'.")
        else:
            raise FileNotFoundError(f"The model file does not exist.")

        self.MODEL_PATH = model_path
        self.yolo_object= YOLO(model_path)

    def predict_frame(self, frame, conf_human = 0.2, conf_hardhat = 0.75):
        classNames = ['human', 'hard_hat', 'no_hard_hat', 'safety_vest', 'forklift', 'transpalet']
        results = self.yolo_object(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_no = int(box.cls[0])   
                class_name = classNames[class_no]

                # if(class_name not in ['yok-baret', 'var-baret']):
                #     continue
            
                if class_name not in ['human', 'hard_hat']:
                    continue


                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
            
            
                if class_name == 'human' and box.conf[0] < conf_human:
                    continue
                elif class_name == 'hard_hat' and box.conf[0] < conf_hardhat:
                    continue
        
                color = (0,255,0)
                if class_name == "hard_hat":
                    color = (0,0,255)
                else:  
                    # Apply blur to the ROI
                    roi = frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (35, 35), 0)
                    frame[y1:y2, x1:x2] = blurred_roi
        

                label = f'{class_name} : {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)

                cv2.putText(frame, label, (x1, y1), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)


                      
                    

def detect_and_update_frame(frame, conf_human = 0.2, conf_hardhat = 0.75):
    global yolo_object_hardhat
    classNames = ['human', 'hard_hat', 'no_hard_hat', 'safety_vest', 'forklift', 'transpalet']
    results = yolo_object_hardhat(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_no = int(box.cls[0])   
            class_name = classNames[class_no]

            # if(class_name not in ['yok-baret', 'var-baret']):
            #     continue
        
            if class_name not in ['human', 'hard_hat']:
                continue


            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
         
        
            if class_name == 'human' and box.conf[0] < conf_human:
                continue
            elif class_name == 'hard_hat' and box.conf[0] < conf_hardhat:
                continue
    
            color = (0,255,0)
            if class_name == "hard_hat":
                color = (0,0,255)
            else:  
                # Apply blur to the ROI
                roi = frame[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (35, 35), 0)
                frame[y1:y2, x1:x2] = blurred_roi
    

            label = f'{class_name} : {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)

            cv2.putText(frame, label, (x1, y1), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

    return frame 
