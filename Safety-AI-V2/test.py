from pprint import pprint

from modules import detector
import cv2

frames = [
    cv2.imread("images/hard_hat.png"),
    cv2.imread("images/no_hard_hat.png"),  
]

detector_object = detector.Detector()

for frame in frames:
    r = detector_object.predict_frame_and_return_detections(frame)
    print("\n\n")
    pprint(r)
