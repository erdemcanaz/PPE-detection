from pprint import pprint
import time

import cv2

from modules.video_feeder import VideoFeeder
from modules.detector import Detector
from modules.memoryless_violation_evaluator import MemorylessViolationEvaluator
from modules.ui_module import UIModule

from scripts.frame_visualizer import FrameVisualizerSimple
from scripts.camera import Camera

frame_visualizer = FrameVisualizerSimple()
video_feeder_object = VideoFeeder()
detector_object = Detector(pose_model_index = 4, hard_hat_model_index = 0, forklift_model_index = 0)
memoryless_violation_evaluator_object = MemorylessViolationEvaluator()
ui_module_object = UIModule()

skipping_second = 2
all_recording_indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

recordings_to_check = [1,2,4,5,6]

while True: #process all recodings
    start_time = time.time()  

    iteration_detection_results = []    
    for recording_index in recordings_to_check:
        video_feeder_object.change_to_video(recording_index)
        frame, ret, NVR_ip, channel, uuid = video_feeder_object.get_current_video_frame() 
           
        detections = detector_object.predict_frame_and_return_detections(frame= frame, camera_uuid= uuid )
        evaluation_results = memoryless_violation_evaluator_object.evaluate_for_violations(detections = detections, camera_uuid = uuid)
        iteration_detection_results.append(evaluation_results)

        video_feeder_object.fast_forward_seconds(skipping_second)     
        
    if not ui_module_object.update_ui_frame(multiple_camera_evaluation_results=iteration_detection_results, window_scale_factor= 0.75, emoji_scale_factor= 1.25, wait_time_ms= 1):
        break      

    end_time = time.time()
    elapsed_time = end_time - start_time   
    print(f"IPS ={str(round(1/elapsed_time,1)):<5},  FPS = {str(round(len(recordings_to_check)/elapsed_time,1)):<5}, skipping second = {skipping_second}", end="\n") #\r
    
  
