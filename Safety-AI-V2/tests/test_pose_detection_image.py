from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-pose.pt')

# Load the image
image = cv2.imread('pose_test_image.jpg')

# Predict with the model
results = model(image, show=True)

# Close the window
cv2.destroyAllWindows()
