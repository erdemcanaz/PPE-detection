import json
import pprint
import cv2
import scripts.video_analyzer as video_analyzer
import scripts.detect_pose as detect_pose
import scripts.detect_hard_hat as detect_hard_hat
import scripts.csv_exporter as csv_exporter
import scripts.object_tracker as object_tracker


def post_process_restriced_area(report_config: dict = None, pre_process_results: list[dict] = None, video_analyzer_object: video_analyzer = None):

    tracking_csv_exporter_object = csv_exporter.CSV_Exporter(
        folder_path=report_config["new_folder_path_dynamic_key"], file_name_wo_extension="post_process_tracking_results")
    pose_detector_object = detect_pose.poseDetector(
        model_path=report_config["post_pose_detection_model_path"])
    object_tracker_object = object_tracker.TrackerSupervisor(
        max_age=5, max_px_distance=200, confidence_threshold=0.5)

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

    all_records = []
    for violation_index, violation_interval in enumerate(violation_intervals):
        object_tracker_object.clear_trackers()
        video_analyzer_object.set_current_seconds(violation_interval[0])
        while video_analyzer_object.get_current_seconds() < violation_interval[1]:
            sampled_frame = video_analyzer_object.get_current_frame()
            pose_detector_object.predict_frame(
                frame=sampled_frame, h_angle=REGION_DATA["CAMERA_H_VIEW_ANGLE"], v_angle=REGION_DATA["CAMERA_V_VIEW_ANGLE"])
            pose_detector_object.approximate_prediction_distance(
                box_condifence_threshold=0.20, distance_threshold=1, transformation_matrices=transformation_matrices)
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
            object_tracker_object.update_trackers_with_detections(
                detections=tracker_detections)

            # informative part, no functional purpose================================
            if report_config["show_video"]:
                pose_detector_object.draw_bounding_boxes(
                    confidence_threshold=0.1, add_blur=True, blur_kernel_size=15)
                pose_detector_object.draw_keypoints_points(
                    confidence_threshold=0.25, DOT_SCALE_FACTOR=1)
                pose_detector_object.draw_upper_body_lines(
                    confidence_threshold=0.1)
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
            for track_record in tracking_record:
                tracking_csv_exporter_object.append_row(track_record)


def post_process_hard_hat(report_config: dict = None, pre_process_results: list[dict] = None, video_analyzer_object: video_analyzer = None):
    hard_hat_csv_exporter_object = csv_exporter.CSV_Exporter(
        folder_path=report_config["new_folder_path_dynamic_key"], file_name_wo_extension="post_process_hard_hat_results")
    hard_hat_detector_object = detect_hard_hat.hardHatDetector(
        model_path=report_config["hard_hat_detection_model_path"])


    #get the seconds when a person(s) is detected
    human_detected_pre_detections = {}
    for detection_dict in pre_process_results:
        if detection_dict["current_second"] in human_detected_pre_detections:
            human_detected_pre_detections[detection_dict["current_second"]].append(detection_dict)     
        else:
            human_detected_pre_detections[detection_dict["current_second"]] = [detection_dict]

    #iterate through the seconds and detect hard hats
    for detection_second, human_predictions in human_detected_pre_detections.items():
        video_analyzer_object.set_current_seconds(detection_second)
        sampled_frame = video_analyzer_object.get_current_frame()
        hard_hat_detector_object.predict_frame(sampled_frame)
        hard_hat_predictions = hard_hat_detector_object.get_prediction_results()

        pprint.pprint(hard_hat_predictions["predictions"])





