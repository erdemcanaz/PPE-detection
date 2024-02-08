import json,pprint
import cv2
import scripts.video_analyzer as video_analyzer
import scripts.detect_pose as detect_pose
import scripts.csv_exporter as csv_exporter
import scripts.object_tracker as object_tracker



def post_process(report_config:dict=None, pre_process_results:list[dict]=None, video_analyzer_object:video_analyzer=None, pose_detector_object:detect_pose=None, csv_exporter_object:csv_exporter=None, transformation_matrices:tuple=None, REGION_DATA:dict=None):
    #==============================INITIALIZE OBJECTS AND CONFIGS===========================
    pose_detector_object = detect_pose.poseDetector( model_path= report_config["post_pose_detection_model_path"])
    object_tracker_object = object_tracker.TrackerSupervisor(max_age = 5, max_px_distance = 200, confidence_threshold = 0.5, speed_attenuation_constant = 1e-6)
    
    r_x1,r_y1,r_x2,r_y2 = REGION_DATA["RESTRICTED_AREA_COORDINATES"]

    violation_seconds = []
    for detection_dict in pre_process_results:
        if detection_dict["is_coordinated_wrt_world_frame"] == True:
            person_x = float(detection_dict["person_x"])
            person_y = float(detection_dict["person_y"])

            if r_x1 < person_x < r_x2 and r_y1 < person_y < r_y2:
                violation_seconds.append(detection_dict["current_second"])

    #for each violation, create an interval of HALF_VIOLATION_TIME seconds before and after the violation
    violation_intervals = []
    for violation_second in violation_seconds:
        should_add_interval = True
        for violation_interval in violation_intervals:
            if violation_interval[0] < violation_second and violation_second < violation_interval[1]:
                violation_interval[0] = min(violation_interval[0], violation_second - report_config["half_violation_time"])
                violation_interval[1] = max(violation_interval[1], violation_second + report_config["half_violation_time"])
                should_add_interval = False
                break
        if should_add_interval:
            violation_intervals.append([violation_second - report_config["half_violation_time"], violation_second + report_config["half_violation_time"]])

            
          

# #####
# #analyze the video for each violation interval for restricted area violation

# all_records = []
# for violation_index, violation_interval in enumerate(violation_intervals):
#     object_tracker_object.clear_trackers()
#     video_analyzer_object.set_current_seconds(violation_interval[0])
#     while video_analyzer_object.get_current_seconds()< violation_interval[1]:
#         sampled_frame = video_analyzer_object.get_current_frame()
#         pose_detector_object.predict_frame(frame = sampled_frame, h_angle= CAMERA_H_ANGLE, v_angle = CAMERA_V_ANGLE)
#         pose_detector_object.approximate_prediction_distance(box_condifence_threshold = 0.20, distance_threshold = 1, transformation_matrices = transformation_matrices)
#         pose_results = pose_detector_object.get_prediction_results()

#         #format prediction results according to tracker

#         tracker_detections = []    
#         for prediction_no, pose_result in enumerate(pose_results["predictions"]):
#             if pose_result["is_coordinated_wrt_world_frame"]:                 
#                 person_x = pose_result["belly_coordinate_wrt_world_frame"][0][0]
#                 person_y = pose_result["belly_coordinate_wrt_world_frame"][1][0]
#                 person_z = pose_result["belly_coordinate_wrt_world_frame"][2][0]

#                 x1,y1,x2,y2 = pose_result["bbox"]
#                 tracker_dict = {
#                     "bbox_center": [(x1+x2)/2, (y1+y2)/2],
#                     "bbox_xyxy": [x1,y1,x2,y2],
#                     "confidence": pose_result["bbox_confidence"],
#                     "person_coordinate": [person_x, person_y, person_z]
#                 }    
#                 tracker_detections.append(tracker_dict)

        
#         object_tracker_object.update_trackers_with_detections(tracker_detections, timestamp = video_analyzer_object.get_current_seconds())

#         #Informative part, no functional purpose================================
#         pose_detector_object.draw_bounding_boxes( confidence_threshold = 0.1, add_blur = True, blur_kernel_size = 15)
#         pose_detector_object.draw_keypoints_points(confidence_threshold = 0.25, DOT_SCALE_FACTOR = 1)
#         pose_detector_object.draw_upper_body_lines(confidence_threshold = 0.1)    
#         object_tracker_object.draw_trackers(sampled_frame)

#         number_of_detections = len(pose_results["predictions"])        
#         cv2.imshow("Frame", sampled_frame)
#         cv2.waitKey(1)
#         print(f"{violation_index}/{len(violation_intervals)} | {video_analyzer_object.get_current_seconds()}s : Frame {video_analyzer_object.get_current_frame_index()}/{video_analyzer_object.get_total_frames()} - {number_of_detections} detections")

#         video_analyzer_object.fast_forward_frames(FRAME_STEP)
    
#     for track_record_id,track_records in object_tracker_object.get_tracker_records().items():
#         for track_record in track_records:
#             csv_exporter.export_to_csv(track_record)
   

# exit()
