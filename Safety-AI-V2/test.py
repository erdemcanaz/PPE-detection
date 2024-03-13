from pprint import pprint
import time

import cv2

from scripts.video_feeder import VideoFeeder
from scripts.frame_visualizer import FrameVisualizerSimple
from modules.detector import Detector
from scripts.camera import Camera

frame_visualizer = FrameVisualizerSimple()
detector_object = Detector(pose_model_index = 4, hard_hat_model_index = 0, forklift_model_index = 0)
video_feeder_object = VideoFeeder()

frames = [
    cv2.imread("images/frame_1.png"),
    cv2.imread("images/frame_2.png"),  
    cv2.imread("images/frame_3.png"),
    cv2.imread("images/frame_4.png"),
    cv2.imread("images/frame_5.png"),
]

for frame in frames:  
    detections = detector_object.predict_frame_and_return_detections(frame= frame, camera_uuid= "7cabf973-f717-44a7-a261-2a3ec7cc610c" )   

    #frame_visualizer.show_frame(frame_name="FrameVisualizer", frame = frame, detections = detections, scale_factor= 0.75)

start_time = time.time()    
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

