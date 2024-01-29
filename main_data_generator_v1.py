import json, pprint,time, random, math, copy, uuid, os, datetime

import cv2
import numpy as np

import scripts.IP_camera.fetch_stream
#import scripts.object_detection.detect_ppe_26_01_2024
#import scripts.object_detection.detect_ppe
#import scripts.object_detection.detect_pose
import scripts.object_detection.detect_ppe_MVP_29_01_2024

def create_grid(frame_details, grid_size=(1, 1)):
    # Determine window size and cell size
    window_height, window_width = 720, 1280  # You can adjust this size
    cell_height, cell_width = window_height // grid_size[0], window_width // grid_size[1]

    frames = []
    for frame_detail in frame_details:
        camera_name = frame_detail[0]
        fame = frame_detail[1]
        timestamp = frame_detail[2]
        time_obj = datetime.datetime.fromtimestamp(timestamp)

        # Extract hour, minute, and second
        hour = time_obj.hour
        minute = time_obj.minute
        second = time_obj.second

        timestamp_human_readable = "{:02d}:{:02d}:{:02d}".format(hour,minute,second)
        camera_text = f"{camera_name} - {timestamp_human_readable}"

        camera_name_added_frame = cv2.putText(fame, camera_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4, cv2.LINE_AA)
        frames.append(camera_name_added_frame)

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
# for camera_id, camera_values in cameras["server_21"]["connected_cameras"].items():
#     if camera_values["status"] == "active":
#         camera_watcher_object = scripts.IP_camera.fetch_stream.IPCameraWatcher(**camera_values)
#         camera_watchers.append(camera_watcher_object)
for camera_id, camera_values in cameras["server_22"]["connected_cameras"].items():
    if camera_values["status"] == "active":
        camera_watcher_object = scripts.IP_camera.fetch_stream.IPCameraWatcher(**camera_values)
        camera_watchers.append(camera_watcher_object)
for camera_id, camera_values in cameras["server_23"]["connected_cameras"].items():
    if camera_values["status"] == "active":
        camera_watcher_object = scripts.IP_camera.fetch_stream.IPCameraWatcher(**camera_values)
        camera_watchers.append(camera_watcher_object)
for camera_id, camera_values in cameras["server_24"]["connected_cameras"].items():
    if camera_values["status"] == "active":
        camera_watcher_object = scripts.IP_camera.fetch_stream.IPCameraWatcher(**camera_values)
        camera_watchers.append(camera_watcher_object)

NUMBER_OF_CAMERAS_TO_WATCH = 9 #you can save images only from the first 9 cameras, otherwise you need to change the save keys
APPLY_OBJECT_DETECTION_MODEL = False
OBJECT_DETECTION_FUNCTION = scripts.object_detection.detect_ppe_MVP_29_01_2024.detect_and_update_frame
SAVE_PATH = "local/saved_images" 
image_counter = 0
DATASET_NAME = "ppe_dataset_v1"
camera_superviser = scripts.IP_camera.fetch_stream.IPcameraSupervisor(camera_watchers, MAX_ACTIVE_STREAMS=NUMBER_OF_CAMERAS_TO_WATCH, MIN_CAMERA_STATUS_DELAY_s= 5,  VERBOSE= True)
camera_superviser.watch_random_cameras([
    "s23-camera 33",
    "s23-camera 32",
    #"s23-camera 23",

    #"s23-camera 25",
    #"s23-camera 16",
    "s23-camera 15",
    "s23-camera 14",

    #"s23-camera 26",
    #"s23-camera 27",
    #"s23-camera 19"
])

if(SAVE_PATH != None):
    r = input(f"Do you want to save the images to path '{SAVE_PATH}'? (yes/no)")
    if(r != "yes"):
        raise Exception("User did not want to save the images to the specified path")
    else:
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

while True:    
   
    camera_superviser.fetch_last_stream()
    fetched_frames = camera_superviser.get_last_fetched_frames_detailed()# [ [camera name, frame, and timestamp], [camera name, frame, and timestamp], ...]
    
    # Create a nxn grid of frames
    if fetched_frames != None:

        original_frames = copy.deepcopy(fetched_frames)

        if APPLY_OBJECT_DETECTION_MODEL:
            for frame in fetched_frames:
                if isinstance(frame[1],np.ndarray):
                    OBJECT_DETECTION_FUNCTION(frame[1])    
                    
        grid_dimension =  math.ceil(math.sqrt(len(fetched_frames)))
        grid = create_grid(fetched_frames, grid_size=(grid_dimension, grid_dimension))
        
        # Display the grid
        cv2.imshow("Camera Grid", grid)        
        date = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
       
        save_keys = [ord(str(i)) for i in range(1,10)]
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press and mask with 0xFF

        if key == ord('q'):  # Compare against the ASCII value of 'q'
            break
        elif key == ord('s'):
            continue   
        elif key == ord('r'):
            camera_superviser.watch_random_cameras([
                "s23-camera 33",
                "s23-camera 32",
                "s23-camera 23",

                "s23-camera 25",
                "s23-camera 16",
                "s23-camera 15",

                "s23-camera 26",
                "s23-camera 27",
                "s23-camera 19"
            ])
        elif key in save_keys:
            uuid_for_frame = uuid.uuid4()

            camera_name = original_frames[key - save_keys[0]][0]
            camera_name = camera_name.replace(" ", "_").replace("-", "_")

            frame = original_frames[key - save_keys[0]][1]
            cv2.imwrite(f"{SAVE_PATH}/secret_{DATASET_NAME}_{camera_name}_{date}_{uuid_for_frame}.jpg", frame)
            image_counter +=1
            print(f"{image_counter}: Image saved to {SAVE_PATH}/secret_data_{date}_{uuid_for_frame}.jpg")

            if image_counter % 50 == 0:
                camera_superviser.watch_random_cameras([])
      
cv2.destroyAllWindows()

