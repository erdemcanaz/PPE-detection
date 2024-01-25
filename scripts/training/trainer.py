from ultralytics import YOLO
import torch

# Option 1: Load a pretrained model
model = YOLO('yolov8n.pt')

# Option 2: Build from YAML and transfer pretrained weights
model = YOLO('yolov8n.yaml').load('C:\\Users\\Levovo20x\\Documents\\GitHub\\PPE-detection\\scripts\\training\\training_results\\ppe_detection_25_01_2024_19-01\\weights\\last.pt')
#model = YOLO('yolov8n.yaml')

# # Resume training from a checkpoint
# model = YOLO("last_saved_model.pt")
# results = model.train(epochs=200, save_period=10, resume=True)

RUN_ON_CUDA = False
if RUN_ON_CUDA and torch.cuda.is_available():
    model.to('cuda')
    print("GPU (CUDA) is detected. Training will be done on GPU.")
else:
    r = input("GPU (CUDA) is not detected or prefered. Should continue with CPU? (y/n):")
    if r != 'y':
        print("Exiting...")
        exit()

# Train the model
experiment = input("Enter the name of your experiment: ")
save_dir = input("Enter the path to your save directory: ")
yaml_file = input("Enter the path to your yaml file: ")

results = model.train(
    data=yaml_file,
    classes = [1,2,4],
    epochs=100, 
    save_dir=save_dir,
    project=save_dir,
    name=experiment,
    imgsz=640,
    save_period = 10)


