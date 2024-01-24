import json, pprint,time, random, math, copy, uuid, os

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


camera_superviser = scripts.IP_camera.fetch_stream.IPcameraSupervisor(camera_watchers, MAX_ACTIVE_STREAMS=4, MIN_CAMERA_STATUS_DELAY_s= 10,  VERBOSE= True)
camera_superviser.watch_random_cameras([
    "s3-camera 50",    
])

APPLY_OBJECT_DETECTION_MODEL = True
OBJECT_DETECTION_FUNCTION = scripts.object_detection.detect_pose.detect_and_update_frame
SAVE_PATH = "local/saved_images" 
if(SAVE_PATH != None):
    r = input(f"Do you want to save the images to path '{SAVE_PATH}'? (yes/no)")
    if(r != "yes"):
        raise Exception("User did not want to save the images to the specified path")
    else:
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

while True:    
    camera_superviser.fetch_last_stream()
    fetched_frames = camera_superviser.get_last_fetched_frames_simple()    
    
    # Create a nxn grid of frames
    if fetched_frames != None:


        original_frames = copy.deepcopy(fetched_frames)

        if APPLY_OBJECT_DETECTION_MODEL:
            for frame in fetched_frames:
                if isinstance(frame,np.ndarray):
                    OBJECT_DETECTION_FUNCTION(frame, confidence_threshold = 0.2)    
                    
        grid_dimension =  math.ceil(math.sqrt(len(fetched_frames)))
        grid = create_grid(fetched_frames, grid_size=(grid_dimension, grid_dimension))
        
        # Display the grid
        cv2.imshow("Camera Grid", grid)        
        date = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
       

        save_keys = [ord(str(i)) for i in range(1,10)]
        key = cv2.waitKey() & 0xFF  # Wait for a key press and mask with 0xFF

        if key == ord('q'):  # Compare against the ASCII value of 'q'
            break
        elif key == ord('s'):
            continue   
        elif key == ord('r'):
            camera_superviser.watch_random_cameras(["s3-camera 50"])
        elif key in save_keys:
            uuid_for_frame = uuid.uuid4()
            cv2.imwrite(f"{SAVE_PATH}/secret_data_{date}_{uuid_for_frame}.jpg", original_frames[key - save_keys[0]])
      
cv2.destroyAllWindows()

