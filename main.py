import json

import cv2
import numpy as np

import scripts.IP_camera.fetch_stream
import scripts.object_detection.detect_ppe


#import the json file that contains the camera information password, username, ip address, and stream path
with open('secret_camera_info.json') as file:
    cameras = json.load(file)

cam_48_watcher = scripts.IP_camera.fetch_stream.IPCameraWatcher(**cameras["cam_48"])
cam_48_watcher.start_watching()

cam_47_watcher = scripts.IP_camera.fetch_stream.IPCameraWatcher(**cameras["cam_47"])
cam_47_watcher.start_watching()

def stack_frames(frames, rows, cols):
    if len(frames) != rows * cols:
        raise ValueError("Number of frames does not match the specified grid dimensions")

    # Split the frames into rows
    frame_rows = [frames[i * cols:(i + 1) * cols] for i in range(rows)]

    # Horizontally stack frames in each row
    stacked_rows = [np.hstack(row) for row in frame_rows]

    # Vertically stack the rows
    return np.vstack(stacked_rows)

while True:
    # Fetch the latest frames from each camera
    frames = [
        cam_48_watcher.get_latest_frame(),   
        cam_47_watcher.get_latest_frame()     
    ]

    # Remove any None frames
    frames = [f for f in frames if f is not None]
    if len(frames) == 0:
        continue

    detected_frames = []   

    for frame in frames:
        detected_frames.append(scripts.object_detection.detect_ppe.detect_and_update_frame(frame , confidence_threshold = 0.75)) 
   
    resized_frames = [cv2.resize(f, (800, 600)) for f in detected_frames]
    combined_frame = stack_frames(resized_frames, 1 , len(detected_frames))
    cv2.imshow('Combined Camera Frames', combined_frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

cv2.destroyAllWindows()