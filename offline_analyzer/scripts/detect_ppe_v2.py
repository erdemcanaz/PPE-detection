from datetime import datetime
from ultralytics import YOLO
import cv2
import math

model = YOLO("C:\\Users\\Levovo20x\\Documents\GitHub\\PPE-detection\\offline_analyzer\\scripts\\yolo_models\\secret_ppe_v2.pt")

def detect_ppe_v2(frame):

    classNames = ['human', 'hard_hat', 'no_hard_hat', 'safety_vest', 'forklift', 'transpalet']

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
            label = f'{class_name}: {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            color = (0,0,0)
            if class_name == 'hard_hat':
                color = (0, 255, 0)
                if conf <0.75:
                    color = (50,50,50)

            elif class_name == "no_hard_hat":
                color = (0, 0, 255)
                if conf <0.3:
                    color = (50,50,50)

            if class_name not in ["hard_hat", "no_hard_hat"]:
                continue                
           

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return frame