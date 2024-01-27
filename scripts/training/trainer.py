from ultralytics import YOLO
import torch


def main():
    torch.cuda.set_device(0) # Set to your desired GPU number

    # Option 1: Load a pretrained model
    model = YOLO('yolov8n.pt')

    # Option 2: Build from YAML and transfer pretrained weights
    model = YOLO('yolov8n.yaml').load('C:\\Users\\Levovo20x\\Documents\\GitHub\\PPE-detection\\scripts\\object_detection\\models\\ppe_26_01_2024.pt')

    #model = YOLO('yolov8n.yaml')

    # Resume training from a checkpoint
    # model = YOLO("last_saved_model.pt")
    # results = model.train(epochs=200, save_period=10, resume=True)

    RUN_ON_CUDA = True
    if RUN_ON_CUDA and torch.cuda.is_available():
        model.to('cuda')
        print("GPU (CUDA) is detected. Training will be done on GPU.")
    else:
        r = input("GPU (CUDA) is not detected or prefered. Should continue with CPU? (y/n):")
        if r != 'y':
            print("Exiting...")
            exit()

    #Train the model
    experiment = input("Enter the name of your experiment: ")
    save_dir = input("Enter the path to your save directory: ")
    yaml_file = input("Enter the path to your yaml file: ")

    model.train(
        data=yaml_file,
        classes = [1,2,4],
        epochs=10, 
        save_dir=save_dir,
        project=save_dir,
        name=experiment,
        imgsz=640,
        save_period = 10,
        batch = 8,
        plots = True
    )

if __name__ == '__main__':
    # Necessary for Windows. If the script is frozen, 'freeze_support()' should be omitted.
    main()


