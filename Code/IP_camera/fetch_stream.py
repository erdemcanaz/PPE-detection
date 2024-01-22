import cv2
import numpy as np
import pprint
import threading

def _display_single_ip_camera_stream(username = None, password= None, ip_address = None, stream_path = None, VERBOSE = False):

    if username == None or password == None or ip_address == None or stream_path == None:
        print("Error: username, password, ip_address, or stream_path is None")
        return None

    # Replace with your camera's URL and credentials
    url = f'rtsp://{username}:{password}@{ip_address}/{stream_path}'
    if(VERBOSE):print(f'Using URL: {url}')
    
    # Set up a video capture object
    cap = cv2.VideoCapture(url)

    # Read from the video capture in a loop
    while True:
        ret, frame = cap.read()
        if ret:
            # Process the frame (e.g., display it)
            cv2.imshow('IP Camera Stream', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the capture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def _capture_single_frame_from_ip_camera(camera_name = "No-Name", username = None, password= None, ip_address = None, stream_path = None, VERBOSE = False):
    if username == None or password == None or ip_address == None or stream_path == None:
        print("Error: username, password, ip_address, or stream_path is None")
        return None
    
    # Form the camera's URL
    url = f'rtsp://{username}:{password}@{ip_address}/{stream_path}'

    if VERBOSE:print(f'{camera_name}: fetching single frame from {url}')
    
    # Set up a video capture object
    cap = cv2.VideoCapture(url)

    # Capture a single frame
    ret, frame = cap.read()
    if ret:
        # You can process the frame here
        # For example, you can display it:
        cv2.imshow('IP Camera Frame', frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

    # Release the capture object
    cap.release()

    return frame

