from datetime import datetime
from ultralytics import YOLO
import cv2
import math

############################################################################################################
#THIS FUNCTION IS A MODIFIED VERSION OF THE FUNCTION FROM THE FOLLOWING GITHUB REPOSITORY:
#The licence is not specified in the repository. However, the yolo model used in his code is from the ultralytics
#repository, which is under the GPL-3.0 license. Therefore, this code is also inherits the licence even though not
#explicitly given.
#https://github.com/KaedKazuha/Personal-Protective-Equipment-Detection-Yolov8/blob/master/YOLO_Video.py
############################################################################################################

def detect_ppe_v2(frame):

    model = YOLO("C:\\Users\\Levovo20x\\Documents\GitHub\\PPE-detection\\offline_analyzer\\scripts\\yolo_models\\secret_ppe_v2.pt")
    classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-hardhat',
                  'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest',
                  'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
                  'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

    # Initialize variables
    start_time = datetime.now()
    detection_results = []

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            color = (0,0,0)
            if class_name == 'Hardhat':
                color = (0, 204, 255)
            elif class_name == "Gloves":
                color = (222, 82, 175)
            elif class_name == "NO-hardhat":
                color = (0, 100, 150)
            elif class_name == "Mask":
                color = (0, 180, 255)
            elif class_name == "NO-Safety Vest":
                color = (0, 230, 200)
            elif class_name == "Safety Vest":
                color = (0, 266, 280)
            else:
                color = (85, 45, 255)

            if class_name not in ["Hardhat", "NO-hardhat", "NO-Safety Vest", "Safety Vest"]:
                continue

            if conf > 0.3:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return frame