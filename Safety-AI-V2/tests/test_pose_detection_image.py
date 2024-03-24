from ultralytics import YOLO
import cv2
import pprint

yolo_models = [
    'yolov8n-pose.pt',
    'yolov8s-pose.pt',
    'yolov8m-pose.pt',
    'yolov8l-pose.pt',
    'yolov8x-pose.pt',
    'yolov8x-pose-p6.pt',

]

pprint.pprint(yolo_models)

which_model_index = input("Which model do you want to use? (index):")
which_model_index = int(which_model_index)
which_model = yolo_models[which_model_index]

model = YOLO(which_model)

# Load the image
image = cv2.imread('pose_test_image.jpg')

# Predict with the model
results = model(image, show=True)

# Close the window
cv2.destroyAllWindows()
