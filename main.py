import json, pprint

import cv2
import numpy as np

import scripts.IP_camera.fetch_stream
import scripts.object_detection.detect_ppe


#import the json file that contains the camera information password, username, ip address, and stream path
with open('secret_camera_info.json') as file:
    cameras = json.load(file)

camera_watchers = []

for camera_id, camera_values in cameras["server_3"]["connected_cameras"].items():
    camera_watcher_object = scripts.IP_camera.fetch_stream.IPCameraWatcher(
        camera_name= camera_values["camera_name"],
        camera_information= camera_values["camera_information"],
        username= camera_values["username"],
        password= camera_values["password"],
        ip_address= camera_values["ip_address"],
        stream_path= camera_values["stream_path"],
        frame_width= camera_values["frame_width"],
        frame_height= camera_values["frame_height"],          
        VERBOSE=True
    )

    camera_watchers.append(camera_watcher_object)

print(camera_watchers)
print(camera_watchers[0])

exit()

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