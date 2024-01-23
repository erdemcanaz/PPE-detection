import json, pprint,time, random

import cv2
import numpy as np

import scripts.IP_camera.fetch_stream
import scripts.object_detection.detect_ppe
import scripts.object_detection.detect_pose

def create_grid(frames, grid_size=(3, 3)):
    # Determine window size and cell size
    window_height, window_width = 800, 1200  # You can adjust this size
    cell_height, cell_width = window_height // grid_size[0], window_width // grid_size[1]

    # Resize frames to fit cell size
    resized_frames = [cv2.resize(frame, (cell_width, cell_height)) for frame in frames]

    # Fill with black frames if not enough frames
    while len(resized_frames) < grid_size[0] * grid_size[1]:
        resized_frames.append(np.zeros((cell_height, cell_width, 3), dtype=np.uint8))

    # Create rows for grid
    rows = [np.hstack(resized_frames[i:i+grid_size[1]]) for i in range(0, len(resized_frames), grid_size[1])]

    # Concatenate rows to create the grid
    grid = np.vstack(rows)
    return grid

#import the json file that contains the camera information password, username, ip address, and stream path
with open('secret_camera_info.json') as file:
    cameras = json.load(file)

camera_watchers = []
for camera_id, camera_values in cameras["server_3"]["connected_cameras"].items():
    camera_watcher_object = scripts.IP_camera.fetch_stream.IPCameraWatcher(**camera_values)
    camera_watchers.append(camera_watcher_object)


camera_superviser = scripts.IP_camera.fetch_stream.IPcameraSupervisor(camera_watchers, MAX_ACTIVE_STREAMS=9, MIN_CAMERA_STATUS_DELAY_s= 10,  VERBOSE= True)
camera_superviser.watch_random_cameras()

while True:    
    camera_superviser.fetch_last_stream()
    fetched_frames = camera_superviser.get_last_fetched_frames_simple()
    
    
    # Create a 3x3 grid of frames
    if fetched_frames != None:

        for frame in fetched_frames:
            if isinstance(frame,np.ndarray):
                scripts.object_detection.detect_pose.detect_and_update_frame(frame, confidence_threshold = 0.2)
                print("pose detected")

        grid = create_grid(fetched_frames, grid_size=(3, 3))

        # Display the grid
        cv2.imshow("Camera Grid", grid)        
       
        key = cv2.waitKey() & 0xFF  # Wait for a key press and mask with 0xFF
        if key == ord('q'):  # Compare against the ASCII value of 'q'
            break
        elif key == ord('s'):
            continue

cv2.destroyAllWindows()

