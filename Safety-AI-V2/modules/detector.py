from scripts.safety_equipment_detectors import hardHatDetector
from scripts.vehicle_detectors import forkliftDetector


class Detector():
    def __init__(self) -> None:
        self.hard_hat_detector_object = hardHatDetector(model_path = "yolo_models/hard_hat_detector.pt")
        self.forklift_detector_object = forkliftDetector(model_path = "yolo_models/forklift_detector.pt")

        self.all_predictions = {
            "safety_equipment_detections":[],
            "human_detections":[], 
            "vehicle_detections":[]
        }

    def predict_frame_and_return_detections(self, frame) -> dict:
        
        #apply pose detection
        
        #apply hardhat detection
        self.all_predictions["safety_equipment_detections"] = self.hard_hat_detector_object.predict_frame_and_return_detections(frame)
        self.all_predictions["vehicle_detections"] = self.forklift_detector_object.predict_frame_and_return_detections(frame)

        return self.all_predictions



