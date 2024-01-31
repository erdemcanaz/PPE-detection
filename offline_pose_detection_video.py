from ultralytics import YOLO
from scripts.object_detection.detect_pose import poseDetector as poseDetector
from scripts.object_detection.detect_hard_hat_29_01_2024 import detect_and_update_frame as detect_and_update_frame
import cv2,math,time,os
import sys
import numpy as np

#==============================
pose_model_path = input("Enter the path to your pose detection model: ")
hard_hat_model_path = input("Enter the path to your hard hat detection model: ")

video_path = input("Enter the path to your video: ")
skip_frames = 7

frame2_path = input("Enter the path to your frame2: ")
person_icon_path = "person_icon.png"
#==============================

pose_detector = poseDetector(pose_model_path)
#TODO: hard_hat_detector = hardHatDetector(hard_hat_model_path)

def detect_from_video(video_path = None, skip_frames = 1):

    cap = cv2.VideoCapture(video_path)
    # V = Aw + C     
    A = np.array(
            [
                [ 0.968747,  -1.038660,  -0.752219],
                [ 0.509982,  0.468733, -1.511582],
                [0.808049, 0.326012, 0.220327]
            ]
        )
    C = np.array(
            [
                [-0.648465],
                [-1.303088], 
                [0.189937]
            ]
       )

    frame_counter = 0 
    while True:
        ret, frame = cap.read()
        frame_counter +=1

        if frame_counter % skip_frames !=0:
            continue

        pose_detector.predict_frame(frame, h_angle= 105.5, v_angle= 57.5)
        pose_detector.approximate_prediction_distance(box_condifence_threshold=0.5, transformation_matrices = [A,C])
        #detect_and_update_frame(frame, conf_human = 0.2, conf_hardhat = 0.75)

        pose_detector.draw_bounding_boxes(confidence_threshold=0.25, add_blur = True, blur_kernel_size = 15)
        pose_detector.draw_keypoints_points()
        pose_detector.draw_upper_body_lines()
        #pose_detector.draw_grid()

        #Resize the frame while maintaining aspect ratio
        width = 600  # desired width
        height = int(frame.shape[0] * width / frame.shape[1])  # calculate height based on aspect ratio
        frame = cv2.resize(frame, (width, height))

        frame2 = cv2.imread(frame2_path)
        frame2 = cv2.resize(frame2, (width, height))

        person_icon = cv2.imread(person_icon_path)
        person_icon_resized = cv2.resize(person_icon, (int(width/10), int(height/10)))

        world_coordinates = pose_detector.get_world_coordinates()
        
        for i in range(len(world_coordinates)):
            x_m = world_coordinates[i][0] + 0.5
            y_m = world_coordinates[i][1] + 0.5
            
            x_px = width-int((x_m/10) * width*0.9)
            y_px = int((y_m/10) * width*0.9)

             # Calculate the top-left corner of the icon placement
            top_left_x = x_px - person_icon_resized.shape[1] // 2
            top_left_y = y_px - person_icon_resized.shape[0] // 2

            # Overlay person_icon_resized onto frame2
            for j in range(person_icon_resized.shape[0]):
                for k in range(person_icon_resized.shape[1]):
                    if 0 <= top_left_y + j < height and 0 <= top_left_x + k < width:  # Check if indices are within frame bounds                        
                            frame2[top_left_y + j, top_left_x + k] = person_icon_resized[j, k]


            #center_coordinates = (x_px, y_px)
            #cv2.circle(frame2, center_coordinates, 5, (0, 0, 0), -1)

        combined_frame = np.hstack((frame, frame2))

        # Display the combined image in full screen
        cv2.imshow('Combined Video', combined_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
        detect_from_video(video_path = video_path, skip_frames= skip_frames)

        
