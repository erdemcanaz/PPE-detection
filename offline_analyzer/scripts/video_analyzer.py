import mimetypes, os, math
import cv2
import datetime

class videoAnalyzer:

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

    def import_video(self, video_path: str, video_start_date: datetime.datetime) -> None:
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
    
    def get_current_seconds(self) -> int:
        return self.current_frame_index/self.VIDEO_FPS

    def set_current_frame_index(self, frame_index: int) -> None:
        if frame_index < 0 or frame_index > self.VIDEO_FRAME_COUNT:
            raise ValueError("Invalid frame index")
        self.current_frame_index = frame_index
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
   
    def safe_fast_forward_seconds(self, forwarding_seconds: int) -> bool:
        if forwarding_seconds < 0:
            raise ValueError("Seconds must be positive")
        
        number_of_frames_to_forward = math.floor(forwarding_seconds * self.VIDEO_FPS)
        if (self.current_frame_index + number_of_frames_to_forward) > self.VIDEO_FRAME_COUNT:
            return False
        return True
    
    def fast_forward_seconds(self, forwarding_seconds: int) -> None:
        if forwarding_seconds < 0:
            raise ValueError("Seconds must be positive")       
        number_of_frames_to_forward = math.floor(forwarding_seconds * self.VIDEO_FPS)
        if (self.current_frame_index + number_of_frames_to_forward) > self.VIDEO_FRAME_COUNT:
            if self.current_frame_index == self.VIDEO_FRAME_COUNT:
                return False
            else:
                self.current_frame_index = self.VIDEO_FRAME_COUNT
                return True
                        
        self.current_frame_index += number_of_frames_to_forward
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return True

    def fast_backward_seconds(self, backward_seconds: int) -> None:
        if backward_seconds < 0:
            raise ValueError("Seconds must be positive")
        number_of_frames_to_backward = math.floor(backward_seconds * self.VIDEO_FPS)
        if (self.current_frame_index - number_of_frames_to_backward) < 0:
            if self.current_frame_index == 0:
                return False
            else:
                self.current_frame_index = 0
                return True
        self.current_frame_index -= number_of_frames_to_backward
        self.video_capture_object.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        return True
    
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
