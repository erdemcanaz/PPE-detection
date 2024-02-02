import os, pprint, json,time
import scripts.video_analyzer as video_analyzer
import scripts.detect_pose as detect_pose
import scripts.data_exporter as data_exporter
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

#===========================DEFINE EXPORTERS================================
csv_export_folder = input("Enter the folder path to the to export the results: ")
csv_export_name = REGION_NAME

csv_exporter = data_exporter.dataExporter(folder_path= csv_export_folder, file_name_wo_extension= csv_export_name, export_extension= ".csv")

#==============================ANALYZE VIDEO================================
sampling_interval_bounds = (0.50, 5) #seconds
interval_increment = 0.5
sampling_interval_seconds = sampling_interval_bounds[0]
min_confidence_to_decrese_interval = 0.5

last_restriced_area_detection_time = 0
min_time_between_restricted_area_detection = 5 #seconds

while video_analyzer_object.fast_forward_seconds(sampling_interval_seconds):

    sampled_frame = video_analyzer_object.get_current_frame()

    pose_detector_object.predict_frame(frame = sampled_frame, h_angle= CAMERA_H_ANGLE, v_angle = CAMERA_V_ANGLE)
    pose_detector_object.approximate_prediction_distance(box_condifence_threshold = 0.20, distance_threshold = 1, transformation_matrices = transformation_matrices)
    pose_results = pose_detector_object.get_prediction_results()
   
    #Adjust sampling interval. If no person is detected, increase the interval. If a person is detected, decrease the interval.
    if pose_detector_object.get_max_confidence_among_results() > min_confidence_to_decrese_interval:        
        should_fast_backward = sampling_interval_seconds > sampling_interval_bounds[0]
        
        current_second = video_analyzer_object.get_current_seconds()
        if should_fast_backward:           
            frame_initial = video_analyzer_object.get_current_frame_index()
            video_analyzer_object.fast_backward_seconds(min(sampling_interval_seconds, current_second))
            frame_final = video_analyzer_object.get_current_frame_index()
            print(f"Video is fast-backward since a person is detected at {current_second:.2f} seconds. Frame {frame_initial} -> Frame {frame_final}")
            sampling_interval_seconds = sampling_interval_bounds[0] #sample more frequently
            continue
        sampling_interval_seconds = sampling_interval_bounds[0] #sample more frequently
        print(f"A person is detected at {current_second:.2f} seconds.")
    else:
        sampling_interval_seconds = min(sampling_interval_seconds + interval_increment, sampling_interval_bounds[1]) #sample less frequently
  
    #Export the results
    for prediction_no, pose_result in enumerate(pose_results["predictions"]):
        if pose_result["is_coordinated_wrt_world_frame"]:                 
            person_x = pose_result["belly_coordinate_wrt_world_frame"][0][0]
            person_y = pose_result["belly_coordinate_wrt_world_frame"][1][0]
            person_z = pose_result["belly_coordinate_wrt_world_frame"][2][0]

            detection_info = {
                "date": video_analyzer_object.get_str_current_date(),
                "video_time": video_analyzer_object.get_str_current_video_time(),
                "prediction_no": prediction_no,
                "confidence": pose_result["bbox_confidence"],
                "person_x": f"{person_x:.2f}",
                "person_y": f"{person_y:.2f}",
                "person_z": f"{person_z:.2f}",
                "restricted_area_violation": "NA",
                "simple_height_violation": "NA",
                "hard_hat_violation": "NA",                        
                "faint_alert_violation": "NA"
            }   

            if IS_RESTRICTED_AREA_APPLIED:
                for restricted_region in RESTRICTED_REGIONS: #ensure that restricted regions do not overlap, otherwise results may be duplicated
                    x1, y1, x2, y2 = restricted_region           
                    if x1 <= person_x <= x2 and y1 <= person_y <= y2:
                        print(f"###########Person detected in restricted area at {video_analyzer_object.get_current_seconds():.2f} seconds")
                        detection_info["restricted_area_violation"]="violated"
                    else:
                        detection_info["restricted_area_violation"]="not_violated"

            if IS_SIMPLE_HEIGHT_ALERT_APPLIED:
                if SIMPLE_HEIGHT_ALERT_HEIGHT < person_z:
                    print(f"###########Person detected at a height of {person_z:.2f} meters at {video_analyzer_object.get_current_seconds():.2f} seconds")
                    detection_info["simple_height_violation"] = "violated"
                else:
                    detection_info["simple_height_violation"] = "not_violated"

            csv_exporter.export_to_csv(detection_info)
    

    #Informative part, no functional purpose================================
    pose_detector_object.draw_keypoints_points(confidence_threshold = 0.25, DOT_SCALE_FACTOR = 1)
    pose_detector_object.draw_upper_body_lines(confidence_threshold = 0.1)    
    pose_detector_object.draw_bounding_boxes( confidence_threshold = 0.1, add_blur = True, blur_kernel_size = 15)
    number_of_detections = len(pose_results["predictions"])    
    
    #Print progress
    frame_index_now = video_analyzer_object.get_current_frame_index()
    total_frame_count = video_analyzer_object.get_total_frames()
    print(f"(%{100*frame_index_now/total_frame_count:.1f}) Int: {sampling_interval_seconds:0.2f}s : Frame {frame_index_now}/{total_frame_count} - {number_of_detections} detections")

    cv2.imshow("Frame", sampled_frame)
    cv2.waitKey(1)


