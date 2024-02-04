import pprint
import scripts.data_importer as data_importer
import scripts.data_analyzer as data_analyzer
import scripts.video_analyzer as video_analyzer
from datetime import datetime, timedelta, timezone
import cv2
import numpy as np
import os
if os.name == 'nt':
    os.system('cls')
    # For Unix/Linux or macOS, use 'clear'
else:
    os.system('clear')
#===========================IMPORT VIDEO=================================
video_analyzer_object = video_analyzer.videoAnalyzer()
video_start_date = datetime(year = 2024, month = 1, day = 31, hour = 8, second = 1, tzinfo= timezone(timedelta(hours=3)))##year, month, day, hour, minute, second, tzinfo
input_video_path = input("Enter the path to the video file: ")
video_analyzer_object.import_video(input_video_path, video_start_date = video_start_date)

#========================DEFINE DATA EXPORTER-IMPORTER OBJECTS=================================
csv_file_path = input("Enter the csv file path to import the results: ")
csv_importer = data_importer.dataImporter(file_path= csv_file_path)

tracking_data = csv_importer.import_csv_as_dict()

tracked_persons = {} # a dict that the tracking id as key and all the related tracking data as value
for tracking_dict in tracking_data:
    person_id = tracking_dict["id"]
    if person_id not in tracked_persons:
        tracked_persons[person_id] = []
    tracked_persons[person_id].append(tracking_dict)

# Calculate likelihood for each person_id
all_likelyhoods = []
for person_id in tracked_persons:
    result_dict = data_analyzer.calculate_direct_passage_likelyhood(person_dict=tracked_persons[person_id], min_data_points=5, min_confidence=0.5, x_border=7.5, x_border_width=5)
    all_likelyhoods.append(result_dict)
#sort the person_dict by timestamp
    
all_likelyhoods_sorted = sorted(all_likelyhoods, key=lambda x: x['likelihood'], reverse=True)

# Print the sorted person_ids

all_frames = []
up_to = min(160, len(all_likelyhoods_sorted))
for track_no, tracking_data in enumerate(all_likelyhoods_sorted[0:up_to]):
    tracking_id = tracking_data["id"]

    start_time = tracking_data["start_time"]
    start_seconds = tracking_data["start_seconds"]

    end_time = tracking_data["end_time"]
    end_seconds = tracking_data["end_seconds"]

    time_stamp = f"{start_time} - {end_time}"
    likelihood = tracking_data["likelihood"]
    if likelihood <0:
        continue

    window_size = max(5, len(tracking_data["x_vals"])//10)
    x_vals = tracking_data["x_vals"]
    x_vals_mean_filter = data_analyzer.apply_mean_filter(x_vals, window_size=window_size)

    y_vals = tracking_data["y_vals"]
    y_vals_mean_filter = data_analyzer.apply_mean_filter(y_vals, window_size=window_size)

    print(f"{track_no}/{len(all_likelyhoods_sorted)}| Tracking ID: {tracking_id} | Likelihood: {likelihood:.2f} | {time_stamp}")
    title = f"{track_no}/{len(all_likelyhoods_sorted)}| Tracking ID: {tracking_id} | Likelihood: {likelihood:.2f} | {time_stamp}"    

    blur_strenght = 45 #must be odd
    frame_walking = data_analyzer.return_xy_tracking_results_frame(x_vals = x_vals_mean_filter, y_vals = y_vals_mean_filter, title=title)
    
    video_analyzer_object.set_current_seconds(start_seconds)
    frame_capture_first = video_analyzer_object.get_current_frame()
    frame_capture_first = cv2.GaussianBlur(frame_capture_first, (blur_strenght, blur_strenght), 0)

    video_analyzer_object.set_current_seconds((start_seconds+end_seconds)/2)
    frame_capture_middle = video_analyzer_object.get_current_frame()
    frame_capture_middle = cv2.GaussianBlur(frame_capture_middle, (blur_strenght, blur_strenght), 0)

    video_analyzer_object.set_current_seconds(end_seconds)
    frame_capture_last = video_analyzer_object.get_current_frame()
    frame_capture_last = cv2.GaussianBlur(frame_capture_last, (blur_strenght, blur_strenght), 0)
    
    #1 walkin path, 3 frame capture
    all_frames.append(frame_walking)
    all_frames.append(frame_capture_first)
    all_frames.append(frame_capture_middle)
    all_frames.append(frame_capture_last)

    # cv2.imshow("Tracking Results", frame)
    # cv2.waitKey(0)


def combine_frames_into_pages(all_frames, rows=4, cols=2, spacing=10, background_color=(0, 0, 0)):
    if not all_frames:
        return []
    
    # Assuming the size of the first frame is the target size for all frames
    frame_height, frame_width = all_frames[0].shape[:2]

    # Calculate the required page width to maintain spacing and A4 aspect ratio
    total_spacing_width = (cols - 1) * spacing
    total_spacing_height = (rows - 1) * spacing
    content_width = frame_width * cols + total_spacing_width
    content_height = frame_height * rows + total_spacing_height

    # Adjust for equal spacing at the top and bottom
    page_width = content_width
    extra_vertical_spacing = 2 * spacing  # Equal spacing at top and bottom
    page_height = content_height + extra_vertical_spacing

    # Ensure the page height respects the A4 aspect ratio as closely as possible
    # Adjust the aspect ratio if necessary based on the content and spacing
    a4_aspect_ratio = 1.414  # A4 paper size ratio
    if page_height / page_width < a4_aspect_ratio:
        page_height = int(page_width * a4_aspect_ratio)

    # Resize frames to have the same dimensions
    resized_frames = [cv2.resize(frame, (frame_width, frame_height)) for frame in all_frames]
    
    pages = []
    
    # Calculate total number of pages needed
    total_pages = len(resized_frames) // (rows * cols) + (1 if len(resized_frames) % (rows * cols) > 0 else 0)
    
    for page in range(total_pages):
        # Create a blank page
        page_img = np.full((page_height, page_width, 3), background_color, dtype=np.uint8)
        
        for index in range(rows * cols):
            frame_index = page * (rows * cols) + index
            if frame_index < len(resized_frames):
                frame = resized_frames[frame_index]
                row, col = divmod(index, cols)
                
                # Calculate the position of the current frame with spacing, including top margin
                y = row * (frame_height + spacing) + spacing
                x = col * (frame_width + spacing)
                
                # Place the frame on the page
                page_img[y:y+frame_height, x:x+frame_width] = frame
        
        pages.append(page_img)
    
    return pages

pages = combine_frames_into_pages(all_frames, rows=4, cols=2,background_color=(255, 255, 255))

#Display each page or save to file
for i, page in enumerate(pages):
    #cv2.imshow(f"Page {i+1}", page)
    #cv2.waitKey(0)
    cv2.imwrite(f"images/page_{i+1}.jpg", page)  # Optionally, save the page to a file
cv2.destroyAllWindows()