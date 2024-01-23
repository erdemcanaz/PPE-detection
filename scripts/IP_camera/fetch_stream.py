import cv2
import numpy as np
import time, threading


class IPcameraSuperior:
    def __init__(self, IPcamera_watcher_objects = []):
        
        pass

class IPCameraWatcher: 
    def __init__(self, camera_name = None, camera_information = None, username = None, password = None, ip_address = None, stream_path = None, frame_width = None, frame_height = None, VERBOSE=False):
        self.camera_name = camera_name
        self.camera_info = camera_information
        self.username = username
        self.password = password
        self.ip_address = ip_address
        self.stream_path = stream_path
        self.width = frame_width
        self.height = frame_height
        self.VERBOSE = VERBOSE
        self.latest_frame = None
        self.latest_frame_timestamp = None
        self.running = False

    def __str__ (self):
        return f'IPCameraWatcher object: {self.camera_name} ({self.ip_address})'
    
    def __repr__ (self):
        return f'IPCameraWatcher object: {self.camera_name} ({self.ip_address})'

    def start_watching(self):
        self.running = True
        self.thread = threading.Thread(target=self._watch_camera)
        self.thread.daemon = True  # Set the thread as a daemon
        self.thread.start()

    def _watch_camera(self):
        url = f'rtsp://{self.username}:{self.password}@{self.ip_address}/{self.stream_path}'  
        cap = cv2.VideoCapture(url)

        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.VERBOSE:print(f'Got a frame from {self._get_formatted_url()}')
                self.latest_frame = frame
                self.latest_frame_timestamp = time.time()
            else:
                break

        cap.release()

    def stop_watching(self):
        self.running = False
        self.thread.join()

    def is_watching(self):
        return self.running

    def get_latest_frame(self):
        return self.latest_frame
    
    def get_latest_frame_timestamp(self):
        return self.latest_frame_timestamp
    
    def _get_formatted_url(self, is_secret = True):
        #if is_secret is True, then the username and password will be replaced with <USERNAME> and <PASSWORD>

        if is_secret:
            return f'rtsp://<USERNAME>:<PASSWORD>@{self.ip_address}'
        else:           
            return f'rtsp://{self.username}:{self.password}@{self.ip_address}/{self.stream_path}'



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

