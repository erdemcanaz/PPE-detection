import os, pprint, json
import scripts.video_analyzer as video_analyzer
import scripts.detect_pose as detect_pose
from datetime import datetime, timedelta, timezone
import cv2
#===========================CLEAR TERMINAL=================================
if os.name == 'nt':
    os.system('cls')
    # For Unix/Linux or macOS, use 'clear'
else:
    os.system('clear')

#===========================IMPORT VIDEO=================================
video_analyzer_object = video_analyzer.videoAnalyzer()
video_start_date = datetime(year = 2024, month = 1, day = 31, hour = 8, second = 1, tzinfo= timezone(timedelta(hours=3)))##year, month, day, hour, minute, second, tzinfo
input_video_path = input("Enter the path to the video file: ")
video_analyzer_object.import_video(input_video_path, video_start_date = video_start_date)

#===========================DEFINE YOLO OBJECTS=================================

pose_detection_model_path = input("Enter the path to the pose detection model: ")
pose_detector_object = detect_pose.poseDetector( model_path= pose_detection_model_path)

#==============================IMPORT REGION================================
region_file_path = input("Enter the path to the region file (JSON): ")
region_data = None
with open(region_file_path, 'r') as file:
    region_data = json.load(file)

CAMERA_H_ANGLE = region_data['CAMERA_H_VIEW_ANGLE']
CAMERA_V_ANGLE = region_data['CAMERA_V_VIEW_ANGLE']
A_MATRIX = region_data['CAMERA_A_MATRIX']
C_MATRIX = region_data['CAMERA_C_MATRIX']
transformation_matrices = (A_MATRIX, C_MATRIX)

#==============================ANALYZE VIDEO================================

sampling_interval_seconds = 15

while video_analyzer_object.fast_forward_seconds(sampling_interval_seconds):

    sampled_frame = video_analyzer_object.get_current_frame()

    pose_detector_object.predict_frame(frame = sampled_frame, h_angle= CAMERA_H_ANGLE, v_angle = CAMERA_V_ANGLE)
    pose_detector_object.approximate_prediction_distance(box_condifence_threshold = 0.75, distance_threshold = 1, transformation_matrices = transformation_matrices)
    pose_detector_object.draw_bounding_boxes()

    pose_results = pose_detector_object.get_prediction_results()
    number_of_detections = len(pose_results["predictions"])
    
    frame_index_now = video_analyzer_object.get_current_frame_index()
    total_frame_count = video_analyzer_object.get_total_frames()

    print(f"(%{100*frame_index_now/total_frame_count:.1f}): Frame {frame_index_now}/{total_frame_count} - {number_of_detections} detections")

    cv2.imshow("Frame", sampled_frame)
    cv2.waitKey(1)