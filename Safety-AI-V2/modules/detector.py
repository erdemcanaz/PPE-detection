from scripts.safety_equipment_detectors import HardHatDetector
from scripts.vehicle_detectors import ForkliftDetector
from scripts.pose_detector import PoseDetector
from scripts.general_objects import Camera

class Detector():
    def __init__(self, camera_object:Camera=None, pose_model_index:int = 2, hard_hat_model_index:int = 0, forklift_model_index:int=0) -> None:
        HARD_HAT_MODEL_PATHS = [
            "yolo_models/hard_hat_detector.pt"  #0
        ]
        FORKLIFT_MODEL_PATHS = [
            "yolo_models/forklift_detector.pt"  #0
        ]
        POSE_MODEL_PATHS = [
            "yolo_models/yolov8n-pose.pt",      #0
            "yolo_models/yolov8s-pose.pt",      #1
            "yolo_models/yolov8m-pose.pt",      #2
            "yolo_models/yolov8l-pose.pt",      #3
            "yolo_models/yolov8x-pose.pt",      #4
            "yolo_models/yolov8x-pose-p6.pt",   #5
        ]

        self.hard_hat_detector_object = HardHatDetector(model_path = HARD_HAT_MODEL_PATHS[hard_hat_model_index] )
        self.forklift_detector_object = ForkliftDetector(model_path = FORKLIFT_MODEL_PATHS[forklift_model_index] )
        self.pose_detector_object = PoseDetector(model_path = POSE_MODEL_PATHS[pose_model_index] )

        self.all_predictions = {
            "human_detections":[], 
            "safety_equipment_detections":[],
            "vehicle_detections":[]
        }

    def predict_frame_and_return_detections(self, frame) -> dict:        
        self.all_predictions["safety_equipment_detections"] = self.hard_hat_detector_object.predict_frame_and_return_detections(frame)
        self.all_predictions["vehicle_detections"] = self.forklift_detector_object.predict_frame_and_return_detections(frame)

        self.pose_detector_object.predict_frame(frame)
        self.pose_detector_object.approximate_prediction_distance()
        self.all_predictions["human_detections"] = self.pose_detector_object.get_prediction_results()
        return self.all_predictions



