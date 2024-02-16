import scripts.detect_pose as detect_pose

def generate_data_points():
    model_path = input("Enter the path to the pose detection model: ")
    video_path = input("Enter the path to the video: ")
    
    pose_detector_object = detect_pose.poseDetector(model_path=model_path)




