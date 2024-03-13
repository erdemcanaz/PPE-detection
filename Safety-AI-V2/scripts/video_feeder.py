import json, mimetypes, os, math
from datetime import datetime, timedelta, timezone
from pprint import pprint

import cv2

class VideoFeeder:

    def __init__(self) -> None:
        #this will automatically initialize the videos to be processed
        VIDEOS_TO_PROCESS_FOLDER_PATH = os.path.join("videos", "videos_to_process")
        self.VIDEO_PATHS = [os.path.join(VIDEOS_TO_PROCESS_FOLDER_PATH, video) for video in os.listdir(VIDEOS_TO_PROCESS_FOLDER_PATH) if (video.endswith(".mp4") or video.endswith(".avi"))]

        #get available camera configurations
        with open('json_files/camera_configs.json') as f:
            camera_configs = json.load(f)["cameras"]

        for video_path in self.VIDEO_PATHS:

            #Example video base name -> NVR-03_XRN-6410RB2_CH014(172.16.0.23)_20240215_080000_085924_ID_0100.avi
            video_base_name  = os.path.basename(video_path)
            splitted_video_base_name = video_base_name.split("_")
            video_channel = splitted_video_base_name[2][0:5]
            vide_NVR_ip = splitted_video_base_name[2][6:17]
            video_YYYYMMDD_str = splitted_video_base_name[3]
            video_start_HHMMSS_str = splitted_video_base_name[4]
            video_end_HHMMSS_str = splitted_video_base_name[5]

            # Create a timezone offset of UTC+3
            tz_offset = timezone(timedelta(hours=3))

            # Parse strings into datetime objects
            video_start_datetime = datetime.strptime(video_YYYYMMDD_str + video_start_HHMMSS_str, '%Y%m%d%H%M%S').replace(tzinfo=tz_offset)
            video_end_datetime = datetime.strptime(video_YYYYMMDD_str + video_end_HHMMSS_str, '%Y%m%d%H%M%S').replace(tzinfo=tz_offset)

            video_basic_info = {
                "video_path": video_path,
                "video_base_name": os.path.basename(video_path),
                "related_camera_uuid": None,
                "video_channel": video_channel,
                "video_NVR_ip": vide_NVR_ip,
                "video_start_datetime": video_start_datetime,
                "video_end_datetime": video_end_datetime,
            }

            is_camera_config_found = False
            for camera_config in camera_configs:
                if camera_config["channel"] == video_channel and camera_config["NVR_ip"] == vide_NVR_ip:
                    is_camera_config_found = True
                    video_basic_info["related_camera_uuid"] = camera_config["uuid"]
                    break
            if not is_camera_config_found:
                raise ValueError(f"Camera configuration not found for the video: {video_base_name}")
            
            #TODO: create VideoRecording object and attach it to this object
            

class VideoRecording:

    def __init__(self)-> None:
        self.VIDEO_PATH = None
        self.MIME_TYPE = None
        self.VIDEO_FRAME_COUNT = None    
        self.VIDEO_WIDTH = None
        self.VIDEO_HEIGHT = None
        self.VIDEO_FPS = None  
        self.VIDEO_START_DATE = None
        self.VIDEO_END_DATE = None
        self.TOTAL_SECONDS = None        

        self.video_capture_object = None
        self.current_frame_index = None    

    def import_video(self, video_path: str, video_start_date: datetime) -> None:
        #mime-type = type "/" [tree "."] subtype ["+" suffix]* [";" parameter];
        #NOTE: this is a regex pattern. Thus may not be the best way to check for the path type. Yet simple one.
        if video_path is None:
            raise ValueError("Video path is not provided")
        
        if not os.path.isfile(video_path):
            raise ValueError("Given video file does not exist")

        # Guess the MIME type of the file based on its extension
        mimetype, _ = mimetypes.guess_type(video_path)
        if mimetype is None:
            raise ValueError("MIME type of the video file is not supported")

        # Check if the MIME type is video
        if not mimetype.startswith('video'):
            raise ValueError("Given file is not a video file")
        
        if video_start_date is None:
            raise ValueError("Video start date is not provided")
        
        self.VIDEO_PATH = video_path
        self.video_capture_object = cv2.VideoCapture(video_path)
        self.MIME_TYPE = mimetype
        self.current_frame_index = 0
        self.FRAME_WIDTH = int(self.video_capture_object.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.FRAME_HEIGHT = int(self.video_capture_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.VIDEO_FRAME_COUNT = int(self.video_capture_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.VIDEO_FPS = self.video_capture_object.get(cv2.CAP_PROP_FPS)
        self.VIDEO_START_DATE = video_start_date
        self.VIDEO_END_DATE = video_start_date + datetime.timedelta(seconds = self.VIDEO_FRAME_COUNT/self.VIDEO_FPS)
        self.TOTAL_SECONDS = self.VIDEO_FRAME_COUNT/self.VIDEO_FPS

        hours, remainder = divmod(self.TOTAL_SECONDS, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("=============================================")
        print(f">{'File name':<20}: {os.path.basename(self.VIDEO_PATH)}")
        print(f">{ 'Video path':<20}: {self.VIDEO_PATH}")
        print(f">{ 'Video resolution':<20}: {self.FRAME_WIDTH}x{self.FRAME_HEIGHT}")
        print(f">{ 'Video frame count':<20}: {self.VIDEO_FRAME_COUNT}")
        print(f">{ 'Video FPS':<20}: {self.VIDEO_FPS:.2f}")
        print(f">{ 'Video duration':<20}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f">{ 'MIME type':<20}: {self.MIME_TYPE}")
        print(f">{ 'Video start date':<20}: {self.VIDEO_START_DATE}")
        print(f">{ 'Video end date':<20}: {self.VIDEO_START_DATE + datetime.timedelta(seconds=self.TOTAL_SECONDS)}")
        print("=============================================")

    def get_current_frame_index(self) -> int:
        return self.current_frame_index
    
    def get_video_duration_in_seconds(self)->int:
        return self.TOTAL_SECONDS
    
    def get_total_frames(self)->int:
        return self.VIDEO_FRAME_COUNT
    
    def get_current_frame(self) -> None:
        ret, frame = self.video_capture_object.read()
        #reading increment the frame index by 1, thus we need to decrement it by 1
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return frame
    
    def get_str_current_date(self) -> str:    
        return self.VIDEO_START_DATE + datetime.timedelta(seconds=self.current_frame_index/self.VIDEO_FPS)    
       
    def get_str_current_video_time(self) -> str:
        seconds_now = self.current_frame_index/self.VIDEO_FPS
        hours, remainder = divmod(self.current_frame_index/self.VIDEO_FPS, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def get_current_seconds(self) -> int:
        return self.current_frame_index/self.VIDEO_FPS

    def set_current_frame_index(self, frame_index: int) -> None:
        if frame_index < 0 or frame_index > self.VIDEO_FRAME_COUNT:
            raise ValueError("Invalid frame index")
        self.current_frame_index = frame_index
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
       
    def fast_forward_seconds(self, forwarding_seconds: int) -> None:
        if forwarding_seconds < 0:
            raise ValueError("Seconds must be positive")       
        number_of_frames_to_forward = math.floor(forwarding_seconds * self.VIDEO_FPS)+1
        if (self.current_frame_index + number_of_frames_to_forward) > self.VIDEO_FRAME_COUNT:
            if self.current_frame_index == self.VIDEO_FRAME_COUNT:
                return False
            else:
                self.current_frame_index = self.VIDEO_FRAME_COUNT
                return True
                        
        self.current_frame_index += number_of_frames_to_forward
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return True
    
    def fast_forward_frames(self, number_of_frames: int) -> None:
        if number_of_frames < 0:
            raise ValueError("Number of frames must be positive")
        if (self.current_frame_index + number_of_frames) > self.VIDEO_FRAME_COUNT:
            if self.current_frame_index == self.VIDEO_FRAME_COUNT:
                return False
            else:
                self.current_frame_index = self.VIDEO_FRAME_COUNT
                return True
        self.current_frame_index += number_of_frames
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return True

    def fast_backward_seconds(self, backward_seconds: int) -> None:
        if backward_seconds < 0:
            raise ValueError("Seconds must be positive")
        number_of_frames_to_backward = math.floor(backward_seconds * self.VIDEO_FPS)+1
        if (self.current_frame_index - number_of_frames_to_backward) < 0:
            if self.current_frame_index == 0:
                return False
            else:
                self.current_frame_index = 0
                return True
        self.current_frame_index -= number_of_frames_to_backward
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return True
    
    def fast_backward_frames(self, number_of_frames: int) -> None:
        if number_of_frames < 0:
            raise ValueError("Number of frames must be positive")
        if (self.current_frame_index - number_of_frames) < 0:
            if self.current_frame_index == 0:
                return False
            else:
                self.current_frame_index = 0
                return True
        self.current_frame_index -= number_of_frames
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return True
    
    def set_current_seconds(self, seconds: float) -> None:
        if seconds < 0 or seconds > self.TOTAL_SECONDS:
            raise ValueError("Invalid seconds")
        self.current_frame_index = math.floor(seconds * self.VIDEO_FPS)
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

    def show_current_frame(self, frame_ratio = 1, close_window_after = False) -> None:
        # should be only used to test the video, has no purpose in the final product
        if frame_ratio < 0 or frame_ratio > 1:
            raise ValueError("Frame ratio must be between 0 and 1")
        
        ret, frame = self.video_capture_object.read()
        frame_to_show = frame
        if frame_ratio != 1:  # Only resize if the ratio is not 1
            new_width = int(frame.shape[1] * frame_ratio)
            new_height = int(frame.shape[0] * frame_ratio)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            frame_to_show = resized_frame

        if ret:
            cv2.imshow("Frame", frame_to_show)
            cv2.waitKey(0)
            if close_window_after:
                cv2.destroyAllWindows()
        else:
            print("End of video")
        
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

    def normalize_x_y(self, x: int, y: int) -> tuple:
        return (x/self.FRAME_WIDTH, y/self.FRAME_HEIGHT)
    
    def normalize_area(self, A:float)->float:
        return A/(self.FRAME_WIDTH*self.FRAME_HEIGHT)