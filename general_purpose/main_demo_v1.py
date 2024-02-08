from ultralytics import YOLO
from scripts.object_detection.detect_pose import poseDetector as poseDetector
import numpy as np
import scripts.IP_camera.fetch_stream
import cv2
import json, time
#==============================
pose_model_path = input("Enter the path to your pose detection model: ")
frame2_path = input("Enter the path to your 2D map: ")
person_icon_path = "person_icon.png"
#==============================

pose_detector = poseDetector(pose_model_path)

with open('secret_camera_info.json') as file:
    cameras = json.load(file)

camera_watchers = []
for camera_id, camera_values in cameras["server_23"]["connected_cameras"].items():
    if camera_values["status"] == "active":
        camera_watcher_object = scripts.IP_camera.fetch_stream.IPCameraWatcher(**camera_values)
        camera_watchers.append(camera_watcher_object)

camera_superviser = scripts.IP_camera.fetch_stream.IPcameraSupervisor(camera_watchers, MAX_ACTIVE_STREAMS=1, MIN_CAMERA_STATUS_DELAY_s= 5,  VERBOSE= False)

camera_superviser.watch_random_cameras([
    "s23-camera 15",
])


def detect_from_IP_stream():
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

    frame2 = None
    last_frame_timestamp = 0
    while True:
        camera_superviser.fetch_last_stream()
        fetched_frames = camera_superviser.get_last_fetched_frames_detailed()# [ [camera name, frame, and timestamp], [camera name, frame, and timestamp], ...]

        for frame_data in fetched_frames:
            #Resize the frame while maintaining aspect ratio
            frame = frame_data[1]
            width = 600  # desired width
            height = int(frame.shape[0] * width / frame.shape[1])  # calculate height based on aspect ratio
            
            if frame_data[2] == last_frame_timestamp:
                frame = cv2.resize(frame, (width, height))       
                combined_frame = np.hstack((frame, frame2))

                # Display the combined image in full screen
                cv2.imshow('Combined Video', combined_frame)    
                continue

            last_frame_timestamp = frame_data[2]

            last_detection = time.time()
            frame2 = cv2.imread(frame2_path)
            frame2 = cv2.resize(frame2, (width, height))
            pose_detector.predict_frame(frame, h_angle= 105.5, v_angle= 57.5)
            pose_detector.approximate_prediction_distance(box_condifence_threshold=0.5, transformation_matrices = [A,C])
            pose_detector.draw_bounding_boxes(confidence_threshold=0.25, add_blur = True, blur_kernel_size = 15)
            pose_detector.draw_keypoints_points()
            pose_detector.draw_upper_body_lines()
            #pose_detector.draw_grid()

            world_coordinates = pose_detector.get_world_coordinates()
        
            for i in range(len(world_coordinates)):
                x_m = world_coordinates[i][0] + 0.5
                y_m = world_coordinates[i][1] + 0.5
                
                x_px = width-int((x_m/10) * width*0.9)
                y_px = int((y_m/10) * width*0.9)

                person_icon = cv2.imread(person_icon_path)
                person_icon_resized = cv2.resize(person_icon, (int(width/10), int(height/10)))

                # Calculate the top-left corner of the icon placement
                top_left_x = x_px - person_icon_resized.shape[1] // 2
                top_left_y = y_px - person_icon_resized.shape[0] // 2

                frame2 = cv2.imread(frame2_path)
                frame2 = cv2.resize(frame2, (width, height))
                
                # Overlay person_icon_resized onto frame2
                for j in range(person_icon_resized.shape[0]):
                    for k in range(person_icon_resized.shape[1]):
                        if 0 <= top_left_y + j < height and 0 <= top_left_x + k < width:  # Check if indices are within frame bounds                        
                                frame2[top_left_y + j, top_left_x + k] = person_icon_resized[j, k]

            frame = cv2.resize(frame, (width, height))       
            combined_frame = np.hstack((frame, frame2))

            # Display the combined image in full screen
            cv2.imshow('Combined Video', combined_frame)               

            # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

if __name__ == "__main__":
        detect_from_IP_stream()

        
