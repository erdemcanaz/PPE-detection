import scripts.detect_pose as detect_pose
import scripts.video_analyzer as video_analyzer
from datetime import datetime, timedelta, timezone
import scripts.find_transformation_coefficients as find_transformation_coefficients
import numpy as np

import cv2

def generate_data_points(start_from_sec = 0):
    model_path = "yolo_models/yolov8x-pose-p6.pt"
    video_path = input("Enter the path to the video: ")

    pose_detector_object = detect_pose.poseDetector(model_path=model_path)
    video_analyzer_object = video_analyzer.videoAnalyzer()
    video_start_date = datetime(year = 2000, month = 3,  day = 27,  hour = 1,  second = 0, tzinfo= timezone(timedelta(hours=3)) )
    video_analyzer_object.import_video(video_path=video_path, video_start_date=video_start_date)

    video_analyzer_object.set_current_seconds(start_from_sec)

    while video_analyzer_object.fast_forward_seconds(1):
        sampled_frame = video_analyzer_object.get_current_frame()
        pose_detector_object.predict_frame(frame=sampled_frame, h_angle=60, v_angle=60)

        A= np.array([[1,0,0],[0,1,0],[0,0,1]])
        B= np.array([[0],[0],[0]])
        transformation_matrices = (A, B)

        pose_detector_object.approximate_prediction_distance(box_condifence_threshold=0.20, distance_threshold=1, shoulders_confidence_threshold= 0.5, transformation_matrices=transformation_matrices)
        pose_results = pose_detector_object.get_prediction_results()
        print(pose_results)
        print(video_analyzer_object.get_current_seconds())

        pose_detector_object.draw_bounding_boxes(    confidence_threshold=0.1, add_blur=False, blur_kernel_size=1)
        pose_detector_object.draw_keypoints_points(  confidence_threshold=0.20, DOT_SCALE_FACTOR=1)
        pose_detector_object.draw_upper_body_lines(  confidence_threshold=0.1)

        cv2.imshow("pre-process", sampled_frame)
        cv2.waitKey(0)
        
def test_matrices(A=None, B=None, start_from_sec = 0):
    model_path = "yolo_models/yolov8m-pose.pt"
    video_path = input("Enter the path to the video: ")

    pose_detector_object = detect_pose.poseDetector(model_path=model_path)
    video_analyzer_object = video_analyzer.videoAnalyzer()
    video_start_date = datetime(year = 2000, month = 3,  day = 27,  hour = 1,  second = 0, tzinfo= timezone(timedelta(hours=3)) )
    video_analyzer_object.import_video(video_path=video_path, video_start_date=video_start_date)

    video_analyzer_object.set_current_seconds(start_from_sec)

    while video_analyzer_object.fast_forward_seconds(1):
        sampled_frame = video_analyzer_object.get_current_frame()
        pose_detector_object.predict_frame(frame=sampled_frame, h_angle=60, v_angle=60)
        
        transformation_matrices = (A, B)

        pose_detector_object.approximate_prediction_distance(box_condifence_threshold=0.20, distance_threshold=1, shoulders_confidence_threshold= 0.5, transformation_matrices=transformation_matrices)
        pose_results = pose_detector_object.get_prediction_results()
        print(pose_results)
        print(video_analyzer_object.get_current_seconds())

        pose_detector_object.draw_bounding_boxes(    confidence_threshold=0.1, add_blur=False, blur_kernel_size=1)
        pose_detector_object.draw_keypoints_points(  confidence_threshold=0.20, DOT_SCALE_FACTOR=1)
        pose_detector_object.draw_upper_body_lines(  confidence_threshold=0.1)

        cv2.imshow("pre-process", sampled_frame)
        cv2.waitKey(0)
        
if __name__ == "__main__":
    number = input("Pick a number:\n (1) get-samples\n (2)calculate-matrices\n (3)test matrices\nYour number:")
    if number == "1":
        generate_data_points(start_from_sec=1722)
    elif number == "2":
        find_transformation_coefficients.calculate_transformation_coefficients()
    elif number == "3":
        A = np.array([[0.593,-0.102,-1.245],[0.087, 0.306,-1.549],[-0.304,0.654,0.680]])
        B = np.array([[-0.996],[-1.239],[0.544]])
        test_matrices(A=A, B=B, start_from_sec=1722)
