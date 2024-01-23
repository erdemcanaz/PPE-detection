import time
import cv2

video_name_with_extension = 'Busy people walking the city streets in London HD Stock Footage.mp4'
video_path = f"videos/{video_name_with_extension}"
save_path = "images/"

def save_frame(video_path, save_path, skip_frames=1, delay_between_frames=0.025):
    """
    Play a video and save frames when 's' key is pressed.

    Args:
    video_path (str): Path to the video file.
    save_path (str): Directory where frames will be saved.
    skip_frames (int): Number of frames to skip between checks for key press.
    delay_between_frames (float): Number of seconds to wait between frames.
    """

    print(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_counter = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Frame', frame)

        # Check for 's' key press to save the frame
        if cv2.waitKey(skip_frames):
            if ord('s') == 0xFF:
                frame_name = f"{save_path}/frame_{frame_counter}.jpg"
                cv2.imwrite(frame_name, frame)
                print(f"Saved {frame_name}")

            elif ord('q') == 0xFF:
                break

            elif ord('d')== 0xFF:
                frame_counter += 50 # set 450: if 30 FPS, then 15 seconds
                if frame_counter >= total_frames:
                    break
            
            elif ord('a')== 0xFF:
                frame_counter -= 450
                if frame_counter < 0:
                    frame_counter = 0


        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

# Usage
save_frame(video_path, save_path, delay_between_frames= 0.25)