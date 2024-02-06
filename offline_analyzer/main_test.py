import scripts.video_analyzer as video_analyzer
import scripts.detect_pose as detect_pose
import scripts.detect_ppe as detect_ppe
import scripts.detect_ppe_v2 as detect_ppe_v2

import cv2
from datetime import datetime, timedelta, timezone
import os, json, pprint

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

ppe_detection_model_path = input("Enter the path to the PPE detection model: ")
ppe_detector_object = detect_ppe.ppeDetector( model_path= ppe_detection_model_path)

#==============================IMPORT REGION================================
region_file_path = input("Enter the path to the region file (JSON): ")
region_data = None
with open(region_file_path, 'r') as file:
    region_data = json.load(file)

REGION_NAME = region_data['REGION_NAME']
CAMERA_H_ANGLE = region_data['CAMERA_H_VIEW_ANGLE']
CAMERA_V_ANGLE = region_data['CAMERA_V_VIEW_ANGLE']
A_MATRIX = region_data['CAMERA_A_MATRIX']
C_MATRIX = region_data['CAMERA_C_MATRIX']

IS_RESTRICTED_AREA_APPLIED = region_data["RULES_APPLIED"]["RESTRICTED_AREA"]
RESTRICTED_REGIONS = region_data["RESTRICTED_AREA_COORDINATES"] 

IS_SIMPLE_HEIGHT_ALERT_APPLIED = region_data["RULES_APPLIED"]["SIMPLE_HEIGHT_ALERT"]
SIMPLE_HEIGHT_ALERT_HEIGHT = region_data["SIMPLE_HEIGHT_ALERT_HEIGHT"]

transformation_matrices = (A_MATRIX, C_MATRIX)

#==============================ANALYZE VIDEO================================

sampling_interval_bounds = 0.5
while video_analyzer_object.fast_forward_seconds(sampling_interval_bounds):
    sampled_frame = video_analyzer_object.get_current_frame()

    pose_detector_object.predict_frame(frame = sampled_frame, h_angle= CAMERA_H_ANGLE, v_angle = CAMERA_V_ANGLE)
    pose_detector_object.approximate_prediction_distance(box_condifence_threshold = 0.35, distance_threshold = 1, transformation_matrices = transformation_matrices)
    pose_results = pose_detector_object.get_prediction_results()

    #pprint.pprint(pose_results)
    #Informative part, no functional purpose================================
    
    #pose_detector_object.draw_keypoints_points(confidence_threshold = 0.25, DOT_SCALE_FACTOR = 3)
    #pose_detector_object.draw_upper_body_lines(confidence_threshold = 0.1)    
    #pose_detector_object.draw_bounding_boxes( confidence_threshold = 0.1, add_blur = True, blur_kernel_size = 15)
    
    #for pose_prediction in pose_results["predictions"]:

    #ppe_detector_object.detect_and_update_frame(sampled_frame, conf_human = 0.2, conf_hardhat = 0.75)

    number_of_detections = len(pose_results["predictions"])   

    cv2.imshow("Frame", sampled_frame)  
    if number_of_detections > 0:
        ppe_detector_object.detect_and_update_frame(sampled_frame, conf_human = 0.2, conf_hardhat = 0.75)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
   
    # else:
    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         break

    # for pose_prediction in pose_results["predictions"]:
    #     x1, y1, x2, y2 = pose_prediction["bbox"]
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #     frame_section = sampled_frame[y1:y2, x1:x2]
    #     ppe_detector_object.detect_and_update_frame(frame_section, conf_human = 0.2, conf_hardhat = 0.75)

    #     cv2.imshow("frame", frame_section)
    #     cv2.waitKey(0)
   
cv2.destroyAllWindows()
