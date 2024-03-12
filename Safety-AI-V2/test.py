from pprint import pprint

from scripts.frame_visualizer import FrameVisualizerSimple

import cv2

from modules import detector
from scripts.camera import Camera

frame_visualizer = FrameVisualizerSimple()
camera_object = Camera(uuid="7cabf973-f717-44a7-a261-2a3ec7cc610c")
detector_object = detector.Detector(pose_model_index = 2, hard_hat_model_index = 0, forklift_model_index = 0)

frames = [
    cv2.imread("images/frame_1.png"),
    cv2.imread("images/frame_2.png"),  
    cv2.imread("images/frame_3.png"),
    cv2.imread("images/frame_4.png"),
    cv2.imread("images/frame_5.png"),
]

for frame in frames:
    detections = detector_object.predict_frame_and_return_detections(frame= frame, camera_object=camera_object )
    frame_visualizer.show_frame(frame_name="FrameVisualizer", frame = frame, detections = detections, scale_factor= 0.75)

#TODO: frame detection post-processing (whether in forklift, whether hard hat violation etc.)