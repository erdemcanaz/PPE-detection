import math,os
import cv2
import time

from ultralytics import YOLO

class hardHatDetector:
    def __init__(self, model_path:str )-> None:          
        self.MODEL_PATH = model_path        
        self.yolo_object = YOLO( self.MODEL_PATH, verbose= False)   
        self.recent_prediction_results = None     

    def fill_prediction_dict_template(self) -> dict:
        empty_prediction_dict = {   
                    "DETECTOR_TYPE":"hardHatDetector",          # which detector made this prediction
                    "frame_shape": [0,0],                       # [0,0], [height , width] in pixels
                    "class_name":"",                            # hard_hat, no_hard_hat
                    "bbox_confidence":0,                        # 0.0 to 1.0
                    "bbox_px":[0,0,0,0],                        # [x1,y1,x2,y2] in pixels
                    "center": [0,0],                            # [x,y] in pixels
        }
        return empty_prediction_dict

    def predict_frame_and_return_detections(self, frame) -> list:
        self.recent_prediction_results = []

        results = self.yolo_object(frame, task = "predict", verbose = False)[0]     
        for i, result in enumerate(results):
            boxes = result.boxes
            
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]
            if box_cls_name not in ["hard_hat", "no_hard_hat"]:
                continue
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxy = boxes.xyxy.cpu().numpy()[0]
            
            prediction_dict_template = self.__PREDICTION_DICT_TEMPLATE()

            prediction_dict_template["frame_shape"] = list(results.orig_shape)
            prediction_dict_template["class_name"] = box_cls_name
            prediction_dict_template["hard_hat_bbox_confidence"] = box_conf
            prediction_dict_template["hard_hat_bbox_px"] = box_xyxy # Bounding box in the format [x1,y1,x2,y2]
            prediction_dict_template["hard_hat_center"] = [ (box_xyxy[0]+box_xyxy[2])/2, (box_xyxy[1]+box_xyxy[3])/2]
            self.recent_prediction_results["predictions"].append(prediction_dict_template)