import os
import scripts.video_analyzer as video_analyzer
from datetime import datetime, timedelta, timezone

#clear the terminal
if os.name == 'nt':
    os.system('cls')
    # For Unix/Linux or macOS, use 'clear'
else:
    os.system('clear')


video_analyzer_object = video_analyzer.videoAnalyzer()
video_start_date = datetime(year = 2024, month = 1, day = 31, hour = 8, second = 1, tzinfo= timezone(timedelta(hours=3)))##year, month, day, hour, minute, second, tzinfo
input_video_path = input("Enter the path to the video file: ")
video_analyzer_object.import_video(input_video_path, video_start_date = video_start_date)

number_of_frames = video_analyzer_object.get_total_frames()
frame_now = 0
while True:
    video_analyzer_object.show_current_frame()
    video_analyzer_object.fast_forward_seconds(15*60)


    