from ultralytics import YOLO
import torch
import cv2,math,time,os
import sys

class pose_detector():
    def __init__(self, model_path):
        self.model_path = model_path
        self.yolo_object = YOLO(model_path)
        
        self.detection_results = {
            "frame_shape": [0,0],
            "speed_results": {
                "preprocess": None,
                "inference": None,
                "postprocess": None
            },
            "predictions":[
                {
                    "bbox": [0,0,0,0], # Bounding box in the format [x1,y1,x2,y2]
                    "keypoints": { # Keypoints are in the format [x,y,confidence]
                        "left_eye": [0,0,0],
                        "right_eye": [0,0,0],
                        "nose": [0,0,0],
                        "left_ear": [0,0,0],
                        "right_ear": [0,0,0],
                        "left_shoulder": [0,0,0],
                        "right_shoulder": [0,0,0],
                        "left_elbow": [0,0,0],
                        "right_elbow": [0,0,0],
                        "left_wrist": [0,0,0],
                        "right_wrist": [0,0,0],
                        "left_hip": [0,0,0],
                        "right_hip": [0,0,0],
                        "left_knee": [0,0,0],
                        "right_knee": [0,0,0],
                        "left_ankle": [0,0,0],
                        "right_ankle": [0,0,0]
                    }
                }
            ]
        }

    def predict_frame(self, frame):
        # Get predictions from the model
        results = self.yolo_object(frame, task = "pose")[0]

        self.detection_results['frame_shape'] = list(results.orig_shape) # Shape of the original image->  [height , width]
        self.detection_results["speed_results"] = results.speed # {'preprocess': None, 'inference': None, 'postprocess': None}

        # Parse the predictions
        
        for i, result in enumerate(results):
            print(f"Result {i}")

            print("Z",result)

            boxes = result.boxes  # Boxes object for bbox outputs     
            box_cls = int(boxes.cls.cpu().numpy()[0])
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxy = boxes.xyxy.cpu().numpy()[0]

            print("ZB",box_cls)
            print("ZC",box_conf)
            print("ZD",box_xyxy)           

            keypoints = result.keypoints  # Keypoints object for pose outputs


        return results

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    model_path = input("Enter the path to the model: ")
    detector = pose_detector(model_path)

    frame = cv2.imread(image_path)
    results = detector.get_prediction_results(frame)





