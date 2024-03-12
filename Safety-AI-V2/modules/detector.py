from scripts.safety_equipment_detectors import hardHatDetector


class Detector():
    def __init__(self) -> None:
        self.hard_hat_detector = hardHatDetector(model_path = "yolo_models/hard_hat_detector.pt")
        self.all_predictions = {
            "hard_hat":[],
            "pose":[],
            "forklift":[]
        }
        
    def predict_frame_and_return_detections(self, frame) -> dict:
        
        #apply pose detection
        
        #apply hardhat detection
        self.all_predictions["hard_hat"] = self.hard_hat_detector.predict_frame_and_return_detections(frame)

        #apply forklift detection

        return self.all_predictions



