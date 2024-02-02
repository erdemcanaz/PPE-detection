import mimetypes, os
import cv2
import datetime

class videoAnalyzer:

    def __init__(self)-> None:
        self.VIDEO_PATH = None
        self.VIDEO_FRAME_COUNT = None    
        self.VIDEO_FPS = None  
        self.VIDEO_START_DATE = None

        self.video_capture_object = None
        self.current_frame_index = None    

    def set_video(self, video_path: str, date: datetime.datetime) -> None:
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
        
        self.VIDEO_PATH = video_path
        self.video_capture_object = cv2.VideoCapture(video_path)
        self.VIDEO_FRAME_COUNT = int(self.video_capture_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.VIDEO_FPS = self.video_capture_object.get(cv2.CAP_PROP_FPS)


        print("=============================================")
        print(f"File name: {os.path.basename(self.VIDEO_PATH)}")
        print(f"Video path: {self.VIDEO_PATH}")
        print(f"Video frame count: {self.VIDEO_FRAME_COUNT}")
        print(f"Video FPS: {self.VIDEO_FPS:.2f}")
        total_seconds = self.VIDEO_FRAME_COUNT/self.VIDEO_FPS
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Video duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"MIME type: {mimetype}")
        print(f"Video start date: {date}")
        print(f"Video end date: {date + datetime.timedelta(seconds = total_seconds)}")
        print("=============================================")



