import pprint, random, time, threading, copy
import cv2  
import numpy as np

class IPcameraSupervisor:
    def __init__(self, IPcamera_watcher_objects = [], MAX_ACTIVE_STREAMS = 5, VERBOSE = False, MIN_CAMERA_STATUS_DELAY_s = 60):
        #IPcamera_watcher_objects is a list of IPcameraWatcher objects
        #MAX_ACTIVE_STREAMS is the maximum number of active streams that can be watched at once
        #VERBOSE is a boolean that determines if the program will print out messages

        self.IPcamera_watcher_objects = IPcamera_watcher_objects
        self.active_camera_objects = [] #list of camera objects that are currently being watched
        self.all_camera_names = [] #list of camera object's names under this supervisor
        self.fetched_frames = {} #{camera_name:{frame:None, timestamp:None}, ...}

        for camera_object in self.IPcamera_watcher_objects:
            self.all_camera_names.append(camera_object.camera_name)

        self.MAX_ACTIVE_STREAMS = MAX_ACTIVE_STREAMS
        self.LAST_TIME_STATUS_UPDATED = 0
        self.MIN_CAMERA_STATUS_DELAY_s = MIN_CAMERA_STATUS_DELAY_s
        self.VERBOSE = VERBOSE  

    def __str__ (self):
        return f'IPcameraSuperior object: {len(self.IPcamera_watcher_objects)} camera(s) are being governed by this object'
            
    def watch_random_cameras(self):     
        random_camera_names = random.sample(self.all_camera_names, self.MAX_ACTIVE_STREAMS)
        self.watch_cameras_by_camera_name(random_camera_names)

    def watch_cameras_by_camera_name(self, activated_camera_names = []):
        if time.time() - self.LAST_TIME_STATUS_UPDATED < self.MIN_CAMERA_STATUS_DELAY_s:
            return None        

        if len(activated_camera_names) == 0:
            raise Exception("Error: camera_names is empty")
        elif len(activated_camera_names) > self.MAX_ACTIVE_STREAMS:
            raise Exception(f"Error: There is a limit to number of cameras watched and you can not exceed it. The maximum number of active streams is {self.MAX_ACTIVE_STREAMS}")
        
        active_camera_objects = []
        for activated_camera_name in activated_camera_names:
            if activated_camera_name not in self.all_camera_names:
                raise Exception(f"Error: {activated_camera_name} is not a camera under the supervise of this superviser object")
            
            for camera_object in self.IPcamera_watcher_objects:
                if camera_object.camera_name == activated_camera_name:
                    active_camera_objects.append(camera_object)
                    break
            
        active_camera_objects.extend(self.active_camera_objects[len(activated_camera_names):])
        self.active_camera_objects = active_camera_objects

        self._update_cameras_watching_status()        

        if(self.VERBOSE):
            print(f"\nWatching {len(self.active_camera_objects)} camera(s):")
            pprint.pprint(self.active_camera_objects)  

    def fetch_last_stream(self):

        if(self.VERBOSE): print("\n fetching last stream")

        counter =0
        for camera_object in self.active_camera_objects:
            frame =camera_object.get_latest_frame()
            timestamp = camera_object.get_latest_frame_timestamp()


            if camera_object.camera_name not in self.fetched_frames.keys():
               self.fetched_frames[camera_object.camera_name] = {"frame":None, "timestamp":0}            
            self.fetched_frames[camera_object.camera_name]["frame"] = frame
            self.fetched_frames[camera_object.camera_name]["timestamp"] = timestamp

            if(time.time() - timestamp < 60):
                counter +=1
        
        if(self.VERBOSE):print(f"    {counter} frames fetched are less than 1 minute old")
    
    def get_last_fetched_frames_simple(self):
        frames = []
        for camera_name, frame_info in self.fetched_frames.items():
            frames.append(frame_info["frame"]) 
        return frames
    
    def _update_cameras_watching_status(self):  
        if self.VERBOSE:print("\ncamera watching status are being updated")
        self.LAST_TIME_STATUS_UPDATED = time.time()
        for camera_watcher_object in self.IPcamera_watcher_objects:
           if camera_watcher_object in self.active_camera_objects:
                if not camera_watcher_object.is_watching():
                    if self.VERBOSE:print(f"    Starting to watch {camera_watcher_object.camera_name}")
                    camera_watcher_object.start_watching()
           else:
                if camera_watcher_object.is_watching():
                    if self.VERBOSE:print(f"    Stopping to watch {camera_watcher_object.camera_name}")
                    camera_watcher_object.stop_watching()

        
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
        self.latest_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        self.latest_frame_timestamp = 0
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

