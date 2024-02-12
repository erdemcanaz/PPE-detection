import json
import pprint
import cv2
import scripts.video_analyzer as video_analyzer
import scripts.detect_pose as detect_pose
import scripts.detect_safety_equipment as detect_safety_equipment
import scripts.csv_dealers as csv_dealers
import scripts.object_tracker as object_tracker


def post_process_restriced_area(report_config: dict = None, pre_process_results: list[dict] = None, video_analyzer_object: video_analyzer = None):
    tracking_csv_exporter_object = csv_dealers.CSV_Exporter(folder_path=report_config["new_folder_path_dynamic_key"], file_name_wo_extension="post_process_tracking_results")
    tracking_sorted_csv_exporter_object = csv_dealers.CSV_Exporter(folder_path=report_config["new_folder_path_dynamic_key"], file_name_wo_extension="sorted_post_process_tracking_results")

    pose_detector_object = detect_pose.poseDetector(model_path=report_config["post_pose_detection_model_path"])
    object_tracker_object = object_tracker.TrackerSupervisor(max_age=5, max_px_distance=200, confidence_threshold=0.5)

    REGION_DATA = None
    with open(report_config["region_info_path"], 'r') as file:
        REGION_DATA = json.load(file)
    transformation_matrices = (
        REGION_DATA['CAMERA_A_MATRIX'], REGION_DATA['CAMERA_C_MATRIX'])
    r_x1, r_y1, r_x2, r_y2 = REGION_DATA["RESTRICTED_AREA_COORDINATES"]

    violation_seconds = []
    for detection_dict in pre_process_results:
        if detection_dict["is_coordinated_wrt_world_frame"] == True:
            person_x = float(detection_dict["person_x"])
            person_y = float(detection_dict["person_y"])
            if r_x1 < person_x < r_x2 and r_y1 < person_y < r_y2:
                violation_seconds.append(detection_dict["current_second"])

    # for each violation, create an interval of HALF_VIOLATION_TIME seconds before and after the violation
    violation_intervals = []
    for violation_second in violation_seconds:
        should_add_interval = True
        for violation_interval in violation_intervals:
            if violation_interval[0] < violation_second and violation_second < violation_interval[1]:
                violation_interval[0] = min(
                    violation_interval[0], violation_second - report_config["half_violation_time"])
                violation_interval[1] = max(
                    violation_interval[1], violation_second + report_config["half_violation_time"])
                should_add_interval = False
                break
        if should_add_interval:
            violation_intervals.append([violation_second - report_config["half_violation_time"],
                                       violation_second + report_config["half_violation_time"]])

    all_rows = []
    all_trackings = {}
    for violation_index, violation_interval in enumerate(violation_intervals):
        object_tracker_object.clear_trackers()
        video_analyzer_object.set_current_seconds(violation_interval[0])
        while video_analyzer_object.get_current_seconds() < violation_interval[1]:
            sampled_frame = video_analyzer_object.get_current_frame()

            pose_detector_object.predict_frame(frame=sampled_frame, h_angle=REGION_DATA["CAMERA_H_VIEW_ANGLE"], v_angle=REGION_DATA["CAMERA_V_VIEW_ANGLE"])
            pose_detector_object.approximate_prediction_distance( box_condifence_threshold=0.20, distance_threshold=1, shoulders_confidence_threshold= 0.8, transformation_matrices=transformation_matrices)
            pose_results = pose_detector_object.get_prediction_results()

            tracker_detections = []
            for prediction_no, pose_result in enumerate(pose_results["predictions"]):
                if pose_result["is_coordinated_wrt_world_frame"]:

                    person_x = pose_result["belly_coordinate_wrt_world_frame"][0][0]
                    person_y = pose_result["belly_coordinate_wrt_world_frame"][1][0]
                    person_z = pose_result["belly_coordinate_wrt_world_frame"][2][0]
                    x1, y1, x2, y2 = pose_result["bbox"]

                    tracker_dict = {
                        "is_coordinated_wrt_world_frame": pose_result["is_coordinated_wrt_world_frame"],

                        "date": video_analyzer_object.get_str_current_date(),
                        "total_frame_count": video_analyzer_object.get_total_frames(),
                        "current_frame_index": video_analyzer_object.get_current_frame_index(),
                        "video_duration_seconds": video_analyzer_object.get_video_duration_in_seconds(),
                        "current_second": video_analyzer_object.get_current_seconds(),

                        "bbox_coordinates": [x1, y1, x2, y2],
                        "box_coordinates_normalized": f"[{video_analyzer_object.normalize_x_y(x1,y1)}, {video_analyzer_object.normalize_x_y(x2,y2)}",
                        "bbox_area": int(pose_result["bbox_pixel_area"]),
                        "bbox_area_normalized": f"{video_analyzer_object.normalize_area(pose_result['bbox_pixel_area']):0.4f}",
                        "video_time": video_analyzer_object.get_str_current_video_time(),
                        "prediction_no": prediction_no,

                        "bbox_confidence": f"{pose_result['bbox_confidence']:0.3f}",
                        "right_shoulder_confidence": f"{pose_result['keypoints']['right_shoulder'][2]:0.3f}",
                        "left_shoulder_confidence": f"{pose_result['keypoints']['left_shoulder'][2]:0.3f}",
                        "right_hip_confidence": f"{pose_result['keypoints']['right_hip'][2]:0.3f}",
                        "left_hip_confidence": f"{pose_result['keypoints']['left_hip'][2]:0.3f}",

                        "person_x": f"{person_x:.3f}",
                        "person_y": f"{person_y:.3f}",
                        "person_z": f"{person_z:.3f}",

                        "tracker_id": "None"
                    }
                    tracker_detections.append(tracker_dict)                                    

            # iterate through all human detections and update the trackers
            object_tracker_object.update_trackers_with_detections(detections=tracker_detections)

            # informative part, no functional purpose================================
            if report_config["show_video"]:
                pose_detector_object.draw_bounding_boxes( confidence_threshold=0.1, add_blur=True, blur_kernel_size=15)
                pose_detector_object.draw_keypoints_points( confidence_threshold=0.25, DOT_SCALE_FACTOR=1)
                pose_detector_object.draw_upper_body_lines( confidence_threshold=0.1)
                object_tracker_object.draw_trackers(sampled_frame)

                cv2.imshow("Post-process - restricted area", sampled_frame)
                cv2.waitKey(1)

            if report_config["verbose"]:
                number_of_detections = len(pose_results["predictions"])
                print(f"{violation_index+1}/{len(violation_intervals)} | {video_analyzer_object.get_str_current_video_time()} - {number_of_detections} detections")

            video_analyzer_object.fast_forward_frames(
                report_config["frame_step"])

        # all violation period is analyzed. Now, we can append the records to the csv file and continue with next interval
        for track_record_id, tracking_record in object_tracker_object.get_tracker_records().items():
            all_trackings[track_record_id] = tracking_record
            for track_record in tracking_record:
                all_rows.append(track_record)
                tracking_csv_exporter_object.append_row(track_record)
    cv2.destroyAllWindows()

    #sort according to the violation score (not calculated yet) 
    
    X_MIN = REGION_DATA["X_LINES"]["x_min"]
    X_THRESHOLD = REGION_DATA["X_LINES"]["x_threshold"]
    X_MAX = REGION_DATA["X_LINES"]["x_max"]   

    sorted_tracks = []
    for track_record_id, tracking_record in all_trackings.items():
        left_side_max_point = 0.3 # (x_threshold - {x})*bbox_conf where x < x_threshold
        right_side_max_point = 0.1 # ({x} - x_threshold)*bbox_conf where x > x_threshold
        for track_record in tracking_record:
            person_x = float(track_record["person_x"])

            bbox_conf = float(track_record["bbox_confidence"])
            right_shoulder_confidence = float(track_record['right_shoulder_confidence'])
            left_shoulder_confidence = float(track_record['right_shoulder_confidence'])
            shoulder_confidence_multiplier = min(right_shoulder_confidence, left_shoulder_confidence)

            if person_x < X_THRESHOLD:
                person_x = max(person_x, X_MIN)
                right_side_point = shoulder_confidence_multiplier*bbox_conf*((X_THRESHOLD - person_x)/(X_THRESHOLD-X_MIN))
                right_side_max_point = max(right_side_max_point, right_side_point)

            else:
                person_x = min(person_x, X_MAX)
                left_side_point = shoulder_confidence_multiplier*bbox_conf*((person_x - X_THRESHOLD)/(X_MAX-X_THRESHOLD))
                left_side_max_point = max(left_side_max_point, left_side_point)

        track_violation_score = left_side_max_point*right_side_max_point # between 0 and 1
        print(f"score:{track_violation_score } | left_side_max_point: {left_side_max_point:.2f} | right_side_max_point: {right_side_max_point:.2f}")

        sorted_track_dict = {
            "track_id": track_record_id,
            "violation_score": track_violation_score,
            "first_frame_index": tracking_record[0]["current_frame_index"],
            "first_frame_time": tracking_record[0]["video_time"],
            "first_frame_date": tracking_record[0]["date"],
            "last_frame_index": tracking_record[-1]["current_frame_index"],
            "last_frame_time": tracking_record[-1]["video_time"],
        }
        sorted_tracks.append(sorted_track_dict)
    sorted_tracks = sorted(sorted_tracks, key=lambda x: x['violation_score'], reverse=True)
    for row in sorted_tracks:
        tracking_sorted_csv_exporter_object.append_row(row)

    return all_rows,sorted_tracks

def post_process_hard_hat(report_config: dict = None, pre_process_results: list[dict] = None, video_analyzer_object: video_analyzer = None):
    hard_hat_csv_exporter_object = csv_dealers.CSV_Exporter( folder_path=report_config["new_folder_path_dynamic_key"], file_name_wo_extension="post_process_safety_equipment_results")
    hard_hat_sorted_csv_exporter_object = csv_dealers.CSV_Exporter( folder_path=report_config["new_folder_path_dynamic_key"], file_name_wo_extension="sorted_post_process_safety_equipment_results")
    hard_hat_detector_object = detect_safety_equipment.safetyEquipmentDetector(model_path=report_config["hard_hat_detection_model_path"])

    #get the frames when a person(s) is detected
    human_detected_pre_detections = {}
    for detection_dict in pre_process_results:
        if detection_dict["current_frame_index"] in human_detected_pre_detections:
            human_detected_pre_detections[detection_dict["current_frame_index"]].append(detection_dict)     
        else:
            human_detected_pre_detections[detection_dict["current_frame_index"]] = [detection_dict]

    #iterate through the frames and detect hard hats
    all_rows = []
    for detection_frame_index, human_predictions in human_detected_pre_detections.items():
        video_analyzer_object.set_current_frame_index(detection_frame_index)
        print(f"\n{video_analyzer_object.get_str_current_video_time()} | Analyzing frame: {detection_frame_index}")

        sampled_frame = video_analyzer_object.get_current_frame()        
        hard_hat_detector_object.predict_frame(sampled_frame)
        hard_hat_predictions = hard_hat_detector_object.get_prediction_results()

        human_predicted_regions = [] # [ [x1,y1,x2,y2,confidence, keypoints], ...]
        for human_prediction in human_predictions:
            x1, y1, x2, y2 = human_prediction["bbox_coordinates"]

            #check if the head is in the frame
            atleast_joints = ["left_eye", "right_eye", "nose", "left_ear", "right_ear"]
            is_head_visible = False
            for joint in atleast_joints:
                if float(human_prediction[joint+"_confidence"]) > 0.1:
                    is_head_visible = True
                    break
            if not is_head_visible:
                continue

            human_predicted_regions.append([x1,y1,x2,y2, human_prediction["bbox_confidence"]])
        
        hard_hat_related_predicted_regions = [] # [ [center_x, center_y, confidence, class_name], ...]
        for hard_hat_prediction in hard_hat_predictions["predictions"]:
            x_center, y_center = hard_hat_prediction["hard_hat_center"]
            hard_hat_related_predicted_regions.append([x_center, y_center, hard_hat_prediction["hard_hat_bbox_confidence"], hard_hat_prediction["class_name"] ])

        check_if_inside = lambda x_center, y_center, x1,y1,x2,y2: x1 < x_center < x2 and y1 < y_center < y2

        for human_prediction in human_predicted_regions:
            kx1, ky1, kx2, ky2 = human_prediction[0:4]
            kx1, ky1, kx2, ky2 = int(kx1), int(ky1), int(kx2), int(ky2)          
            cv2.rectangle(sampled_frame, (kx1,ky1), (kx2,ky2), (0,255,0), 2)

            safety_equipment_row = {
                "date": video_analyzer_object.get_str_current_date(),
                "video_time": video_analyzer_object.get_str_current_video_time(),
                "current_second": video_analyzer_object.get_current_seconds(),
                "frame_index": detection_frame_index,
                "bbox_coordinates": [kx1, ky1, kx2, ky2],
                "bbox_confidence": human_prediction[4],
                "is_safety_equipment_present": False,
                "safety_equipment_confidence": 0,
                "safety_equipment_class": "None",
                "safety_equipment_bbox_center": [0,0],
                "safety_equipment_confidence": 0,
                "violation_score":0                
            }
          
            #check if any protective equipment analyze is performed inside the human bbox
            
            safety_equipment_row["violation_score"] = float(human_prediction[4]) # For the case where there is no safety equipment detected
            for hard_hat_prediction in hard_hat_related_predicted_regions:
                is_hard_hat_center_inside_human_bbox = check_if_inside(hard_hat_prediction[0], hard_hat_prediction[1], kx1, ky1, kx2, ky2)
                if is_hard_hat_center_inside_human_bbox:                    
                    safety_equipment_row["is_safety_equipment_present"] = True
                    safety_equipment_row["safety_equipment_bbox_center"] = hard_hat_prediction[0:2]
                    safety_equipment_row["safety_equipment_confidence"] = hard_hat_prediction[2]
                    safety_equipment_row["safety_equipment_class"] = hard_hat_prediction[3]
                    
                    violation_score = None
                    human_box_confidence = float(human_prediction[4])
                    safety_equipment_confidence = float(safety_equipment_row["safety_equipment_confidence"])
                    if safety_equipment_row["safety_equipment_class"] == "hard_hat":
                        violation_score = human_box_confidence*(1-safety_equipment_confidence)
                    elif safety_equipment_row["safety_equipment_class"] == "no_hard_hat":
                        violation_score = human_box_confidence*safety_equipment_confidence
                    else:
                        print("Unknown safety equipment class")
                        violation_score = -1
                    safety_equipment_row["violation_score"] = violation_score 
                    break
                else:                                      
                    continue
            
            print("\t  violation score:", f"{safety_equipment_row['violation_score']:.2f}")
            hard_hat_csv_exporter_object.append_row(safety_equipment_row)
            all_rows.append(safety_equipment_row)

        if report_config["show_video"]:
            hard_hat_detector_object.draw_predictions()
            cv2.imshow("Post-process - safety equipment (hard hat)", sampled_frame)
            cv2.waitKey(1)    
    cv2.destroyAllWindows()

    all_rows_sorted = sorted(all_rows, key=lambda x: x['violation_score'], reverse=True)
    for row in all_rows_sorted:
        hard_hat_sorted_csv_exporter_object.append_row(row)

    return all_rows




