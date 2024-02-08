from datetime import datetime
import math,os
import cv2
from ultralytics import YOLO

class hardHatDetector:
    def __init__(self, model_path : str ) -> None:
        # Check if the file exists
        if os.path.exists(model_path):
        # Check if the file has the specified extension
            _, file_extension = os.path.splitext(model_path)
            if file_extension != ".pt":
                raise ValueError(f"The model file should have the extension '.pt'.")
        else:
            raise FileNotFoundError(f"The model file does not exist.")

        self.MODEL_PATH = model_path        
        self.yolo_object = YOLO( self.MODEL_PATH, verbose= False)        
        self.prediction_results = None
        self._clear_prediction_results()

    def _clear_prediction_results(self):
        self.prediction_results = {
            "frame":None, #The frame that was predicted
            "time_stamp":0, #time.time() value indicating when this prediction happend
            "frame_shape": [0,0],
            "speed_results": {
                "preprocess": None,
                "inference": None,
                "postprocess": None
            },
            "predictions":[
                # The format of the prediction dictionary is specified in self._PREDICTION_DICT_TEMPLATE()
            ]
        }

    def _PREDICTION_DICT_TEMPLATE(self):
        empty_prediction_dict ={                         
                    "class_index":0,
                    "class_name":"NoName",
                    "bbox_confidence":0,
                    "bbox": [0,0,0,0], # Bounding box in the format [x1,y1,x2,y2]
                    "bbox_pixel_area": 0,
                    "bbox_area_normalized": 0,
            }
        return empty_prediction_dict

    
    def predict_frame(self, frame):
        classNames = ['human', 'hard_hat', 'no_hard_hat', 'safety_vest', 'forklift', 'transpalet']

        # Initialize variables
        detection_results = []
        results = self.yolo_object(frame, stream=True)

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
    
    def draw_hard_hat():


