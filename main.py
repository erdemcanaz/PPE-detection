import json
import cv2
import Code.IP_camera.fetch_stream
import numpy as np

#import the json file that contains the camera information password, username, ip address, and stream path
with open('secret_camera_info.json') as file:
    cameras = json.load(file)

koltuk_ambari_watcher = Code.IP_camera.fetch_stream.IPCameraWatcher(**cameras["koltuk_ambari"])
koltuk_ambari_watcher.start_watching()

aku_alani_sonu_watcher = Code.IP_camera.fetch_stream.IPCameraWatcher(**cameras["aku_alani_sonu"])
aku_alani_sonu_watcher.start_watching()

aku_alani_giris_watcher = Code.IP_camera.fetch_stream.IPCameraWatcher(**cameras["aku_alani_giris"])
aku_alani_giris_watcher.start_watching()

koltuk_ambari_depo_arasi_watcher = Code.IP_camera.fetch_stream.IPCameraWatcher(**cameras["koltuk_ambari_depo_arasi"])
koltuk_ambari_depo_arasi_watcher.start_watching()

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
        koltuk_ambari_watcher.get_latest_frame(),
        aku_alani_sonu_watcher.get_latest_frame(),
        aku_alani_giris_watcher.get_latest_frame(),
        koltuk_ambari_depo_arasi_watcher.get_latest_frame()
    ]

    # Remove any None frames
    frames = [f for f in frames if f is not None]

    if len(frames) == 4:  # Ensure we have 4 frames to display
        resized_frames = [cv2.resize(f, (640, 480)) for f in frames]
        combined_frame = stack_frames(resized_frames, 2, 2)  # Combine frames in a 2x2 grid
        cv2.imshow('Combined Camera Frames', combined_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
            break

cv2.destroyAllWindows()