from pprint import pprint
import cv2

from modules import detector
from scripts.camera import Camera

camera_object = Camera(uuid="7cabf973-f717-44a7-a261-2a3ec7cc610c")
detector_object = detector.Detector(camera_object=camera_object, pose_model_index = 2, hard_hat_model_index = 0, forklift_model_index = 0)

frames = [
    cv2.imread("images/hard_hat.png"),
    cv2.imread("images/no_hard_hat.png"),  
]


for frame in frames:
    r = detector_object.predict_frame_and_return_detections(frame)
    print("\n\n")
    pprint(r)
