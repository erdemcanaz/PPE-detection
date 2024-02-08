import json
import cv2
import scripts.video_analyzer as video_analyzer
import scripts.detect_pose as detect_pose
import scripts.csv_exporter as csv_exporter

def pre_process(video_analyzer_object:video_analyzer=None, report_config:dict=None):

    #==============================INITIALIZE OBJECTS AND CONFIGS===========================

    pose_detector_object = detect_pose.poseDetector( model_path= report_config["pre_pose_detection_model_path"])
    csv_exporter_object = csv_exporter.CSV_Exporter(folder_path= report_config["new_folder_path_dynamic_key"], file_name_wo_extension= "pre_process_results")

    REGION_DATA = None
    with open(report_config["region_info_path"], 'r') as file:
        REGION_DATA = json.load(file)

    transformation_matrices = (REGION_DATA['CAMERA_A_MATRIX'], REGION_DATA['CAMERA_C_MATRIX'])

    #==============================ANALYZE VIDEO================================
    sampling_interval_bounds = (report_config["sampling_interval_bounds"][0], report_config["sampling_interval_bounds"][1]) #seconds
    interval_increment = report_config["sampling_interval_increment"] #seconds
    sampling_interval = sampling_interval_bounds[0]

    start_time_seconds = report_config["start_analyzing_from_second"]["hour"]*3600+report_config["start_analyzing_from_second"]["minute"]*60+report_config["start_analyzing_from_second"]["second"]
    end_time_seconds = report_config["end_analyzing_at_second"]["hour"]*3600+report_config["end_analyzing_at_second"]["minute"]*60+report_config["end_analyzing_at_second"]["second"]
    
    if report_config["analyze_only_specific_interval"]:
        video_analyzer_object.set_current_seconds(start_time_seconds)

    pre_process_results = [] #corresponds to row of the csv file
    while video_analyzer_object.fast_forward_seconds(sampling_interval):#when the video ends, the function returns False
        if report_config["analyze_only_specific_interval"]:
            if video_analyzer_object.get_current_seconds() > end_time_seconds:
                break

        sampled_frame = video_analyzer_object.get_current_frame()

        pose_detector_object.predict_frame(frame = sampled_frame, h_angle= REGION_DATA['CAMERA_H_VIEW_ANGLE'], v_angle = REGION_DATA['CAMERA_V_VIEW_ANGLE'])
        pose_detector_object.approximate_prediction_distance(box_condifence_threshold = 0.20, distance_threshold = 1, transformation_matrices = transformation_matrices)
        pose_results = pose_detector_object.get_prediction_results()

        # Adjust sampling interval
        max_confidence = pose_detector_object.get_max_confidence_among_results()
        current_second = video_analyzer_object.get_current_seconds()
        if max_confidence > report_config["min_confidence_to_decrease_interval"] and sampling_interval > sampling_interval_bounds[0]:
            sampling_interval = sampling_interval_bounds[0]
            if current_second > sampling_interval:
                video_analyzer_object.fast_backward_seconds(min(sampling_interval, current_second))
                print(f"A person is detected at {current_second:.2f} seconds.")
        else:
            sampling_interval = min(sampling_interval + interval_increment, sampling_interval_bounds[1])

        #Export the results
        for prediction_no, pose_result in enumerate(pose_results["predictions"]):
            person_x = pose_result["belly_coordinate_wrt_world_frame"][0][0]
            person_y = pose_result["belly_coordinate_wrt_world_frame"][1][0]
            person_z = pose_result["belly_coordinate_wrt_world_frame"][2][0]

            x1, y1, x2, y2 = pose_result["bbox"]

            detection_info = {
                "is_coordinated_wrt_world_frame": pose_result["is_coordinated_wrt_world_frame"],

                "date": video_analyzer_object.get_str_current_date(),
                "total_frame_count": video_analyzer_object.get_total_frames(),
                "current_frame_index": video_analyzer_object.get_current_frame_index(),
                "video_duration_seconds": video_analyzer_object.get_video_duration_in_seconds(),
                "current_second": video_analyzer_object.get_current_seconds(),

                "bbox_coordinates": [x1,y1,x2,y2],
                "bbox_coordinates_str": f"top_left: {x1}, {y1} bottom_right: {x2}, {y2}",
                "box_coordinates_normalized": f"top_left:{video_analyzer_object.normalize_x_y(x1,y1)} bottom_right:{video_analyzer_object.normalize_x_y(x2,y2)}",
                "bbox_area":int(pose_result["bbox_pixel_area"]),
                "bbox_area_normalized":f"{video_analyzer_object.normalize_area(pose_result['bbox_pixel_area']):0.4f}",
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

                "tracker_id":"None"
            }   
            
            pre_process_results.append(detection_info)
            csv_exporter_object.append_row(detection_info)
        
        #====================================VERBOSE PAR, NO FUNCTIONALITY====================================
        if report_config["verbose"]:
            number_of_detections = len(pose_results["predictions"]) 
            frame_index_now = video_analyzer_object.get_current_frame_index()
            total_frame_count = video_analyzer_object.get_total_frames()
            if report_config["analyze_only_specific_interval"]:
                progress_percentage = (video_analyzer_object.get_current_seconds()-start_time_seconds)/(end_time_seconds-start_time_seconds)*100
                print(f"(%{progress_percentage:.2f}) Int: {sampling_interval:0.2f}s : - {number_of_detections} detections")
            else:
                print(f"(%{100*frame_index_now/total_frame_count:.2f}) Int: {sampling_interval:0.2f}s : - {number_of_detections} detections")

        if report_config["show_video"]:        
            pose_detector_object.draw_bounding_boxes( confidence_threshold = 0.1, add_blur = report_config["apply_video_blur"], blur_kernel_size = report_config["video_blur_kernel_size"])
            pose_detector_object.draw_keypoints_points(confidence_threshold = 0.20, DOT_SCALE_FACTOR = 1)
            pose_detector_object.draw_upper_body_lines(confidence_threshold = 0.1)
            cv2.imshow("pre-process", sampled_frame)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    return video_analyzer_object, pre_process_results, REGION_DATA, transformation_matrices








