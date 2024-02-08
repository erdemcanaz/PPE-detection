import math,os
import cv2
from ultralytics import YOLO
import time

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
                    "bbox_area": 0,
                    "bbox_area_normalized": 0,
                    "is_hard_hat_present": False,
            }
        return empty_prediction_dict

    
    def predict_frame(self, frame) -> None:
        self._clear_prediction_results()

        results = self.yolo_object(frame, task = "predict", verbose = False)[0]
        self.prediction_results['frame'] = frame
        self.prediction_results['frame_shape'] = list(results.orig_shape) # Shape of the original image->  [height , width]
        self.prediction_results["speed_results"] = results.speed # {'preprocess': None, 'inference': None, 'postprocess': None}
        self.prediction_results["time_stamp"] = time.time()

        for i, result in enumerate(results):
            boxes = result.boxes
            
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]
            if box_cls_name not in ["hard_hat", "no_hard_hat"]:
                continue
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxy = boxes.xyxy.cpu().numpy()[0]
            bbox_pixel_area = (box_xyxy[2]-box_xyxy[0])*(box_xyxy[3]-box_xyxy[1])

            result_detection_dict = self._PREDICTION_DICT_TEMPLATE()
            result_detection_dict ["class_index"] = box_cls_no
            result_detection_dict ["class_name"] = box_cls_name
            result_detection_dict ["bbox_confidence"] = box_conf
            result_detection_dict ["bbox"] = box_xyxy # Bounding box in the format [x1,y1,x2,y2]
            result_detection_dict ["bbox_pixel_area"] = bbox_pixel_area

            self.prediction_results["predictions"].append(result_detection_dict)
    
    def get_prediction_results(self):
        return self.prediction_results
    
    def draw_predictions(self):
        frame = self.prediction_results['frame']

        for result in self.prediction_results["predictions"]:
            x1, y1, x2, y2 = result["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            conf = result["bbox_confidence"]
            class_name = result["class_name"]
            color = (0,0,0)
            if class_name == 'hard_hat':
                color = (0, 255, 0)
                if conf <0.75:
                    color = (50,50,50)

            elif class_name == "no_hard_hat":
                color = (0, 0, 255)
                if conf <0.3:
                    color = (50,50,50)

            label = f"{class_name}: {conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)