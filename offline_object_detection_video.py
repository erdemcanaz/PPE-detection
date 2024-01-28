from ultralytics import YOLO
import cv2,math,time,os
import sys

#==============================
source = "video"
webcam_id = 0
image_path = "images\\test.jpg"
skip_frames = 1
confidence_threshold = 0.35

model_path = input("Enter the path to your model: ")
video_path = input("Enter the path to your video: ")
#==============================

yolo_object = YOLO(model_path)

def detect_from_frame(frame):
    results = yolo_object(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100

            if(conf < confidence_threshold):
                continue

            cls = int(box.cls[0])               
            label = f'{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
            cv2.rectangle(frame, (x1, y1), c2, (0,255,0), -1, cv2.LINE_AA)

            cv2.putText(frame, label, (x1, y1), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

    return frame

def detect_from_video(video_path = None, skip_frames = 1):
    global yolo_object

    cap = cv2.VideoCapture(video_path)

    frame_counter = 0 
    while True:
        ret, frame = cap.read()
        frame_counter +=1

        if frame_counter % skip_frames !=0:
            continue

        results = yolo_object(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100

                if(conf < confidence_threshold):
                    continue

                cls = int(box.cls[0])               
                class_name = yolo_object.names[cls]
                label = f'{class_name} - {conf}'

                # Apply blur to the ROI
                roi = frame[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)
                frame[y1:y2, x1:x2] = blurred_roi
            
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

               
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
                cv2.rectangle(frame, (x1, y1), c2, (0,255,0), -1, cv2.LINE_AA)

                cv2.putText(frame, label, (x1, y1), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

        # Display the resized image in full screen
        cv2.imshow('video', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_from_webcam(webcam_id = 0):
    cap = cv2.VideoCapture(webcam_id,  cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        
        # Display the resized image in full screen
        cv2.imshow('webcam', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    if source == "video":
        detect_from_video(video_path = video_path, skip_frames= skip_frames)
    elif source == "webcam":
        detect_from_webcam(webcam_id= webcam_id)
    elif source == "frame":
        frame = cv2.imread(image_path)
        frame = detect_from_frame(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
