from ultralytics import YOLO
from scripts.object_detection.detect_pose import poseDetector as poseDetector
import cv2,math,time,os
import sys


#==============================
model_path = input("Enter the path to your model: ")
video_path = input("Enter the path to your video: ")

skip_frames = 150
#==============================

pose_detector = poseDetector(model_path)

def detect_from_video(video_path = None, skip_frames = 1):
    global yolo_object

    cap = cv2.VideoCapture(video_path)

    frame_counter = 0 
    while True:
        ret, frame = cap.read()
        frame_counter +=1

        if frame_counter % skip_frames !=0:
            continue

        pose_detector.predict_frame(frame, h_angle= 105.5, v_angle= 57.5)

        pose_detector.approximate_prediction_distance(h_view_angle= 105.5, v_view_angle= 57.5)

        pose_detector.draw_all()

        #Resize the frame while maintaining aspect ratio
        width = 1000  # desired width
        height = int(frame.shape[0] * width / frame.shape[1])  # calculate height based on aspect ratio
        frame = cv2.resize(frame, (width, height))

        # Display the resized image in full screen
        cv2.imshow('video', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey() & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
        detect_from_video(video_path = video_path, skip_frames= skip_frames)

        
