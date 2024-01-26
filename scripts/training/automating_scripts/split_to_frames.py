import time, uuid, os
import cv2

video_path = input("Please enter the video path to be split into frames: ")
save_path = input("Plese enter the folder path that frames will be saved")

def save_frame(video_path, save_path, skip_frames=30):
    """
    Play a video and save frames when 's' key is pressed.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_counter = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF  # Correct usage of waitKey with mask for 64-bit systems
        if key == ord('s'):
            uuid_val = uuid.uuid4()
            frame_name = os.path.join(save_path, f"{video_name}_{uuid_val}.jpg")
            cv2.imwrite(frame_name, frame)
            print(f"Saved {frame_name}")

        elif key == ord('q'):
            break

        elif key == ord('d'):
            frame_counter += skip_frames
            if frame_counter >= total_frames:
                frame_counter = total_frames - 1

        elif key == ord('a'):
            frame_counter -= skip_frames
            if frame_counter < 0:
                frame_counter = 0

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

# Usage
save_frame(video_path=video_path, save_path=save_path, skip_frames = 30)
