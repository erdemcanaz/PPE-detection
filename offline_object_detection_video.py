from ultralytics import YOLO
import cv2,math,time,os
import sys

#==============================
source = "video"
webcam_id = 0
image_path = "images\\test.jpg"
skip_frames = 60
confidence_threshold_human = 0.15
confidence_threshold_hardhat = 0.75
image_ratio = 0.8 # 1 = full screen, 0.5 = half screen

model_path = input("Enter the path to your model: ")
video_path = input("Enter the path to your video: ")
#==============================

yolo_object = YOLO(model_path)

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

                cls = int(box.cls[0])               
                class_name = yolo_object.names[cls]
                label = f'{class_name} - {conf}'

                # if class_name == "human" and conf < confidence_threshold_human:
                #     continue
            
                # if class_name == "hard_hat" and conf < confidence_threshold_hardhat:
                #     continue                    
                
                # color = (0,255,0)
                # if class_name == "hard_hat":
                #     

                color = (0,0,255)
                roi = frame[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (55, 55), 0)
                frame[y1:y2, x1:x2] = blurred_roi
            
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

               
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)

                cv2.putText(frame, label, (x1, y1), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

        height, width, layers = frame.shape
        new_h = int(height * image_ratio)
        new_w = int(width * image_ratio)
        frame = cv2.resize(frame, (new_w, new_h))

        # Display the resized image in full screen
        cv2.imshow('video', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
        detect_from_video(video_path = video_path, skip_frames= skip_frames)

        
