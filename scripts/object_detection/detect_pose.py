from ultralytics import YOLO
import torch
import cv2,math,time,os
import time, pprint

from scipy.optimize import minimize
import numpy as np


class poseDetector():
    #TODO verbose=False

    KEYPOINT_NAMES = ["left_eye", "rigt_eye", "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow" ,"right_elbow","left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    
    #approximate distances between the keypoints of a person in meters
    SHOULDER_TO_SHOULDER = 0.36 
    SHOULDER_TO_HIP = 0.48 
    SHOULDER_TO_COUNTER_HIP = 0.53  
    SHOULDER_TO_ELBOW = 0.26

    def __init__(self, model_path = "C:\\Users\\Levovo20x\\Documents\\GitHub\\PPE-detection\\scripts\\object_detection\\models\\secret_yolov8x-pose-p6.pt" ):
        self.model_path = model_path
        self.yolo_object = YOLO(model_path)
        
        self.prediction_results = {
            "frame":None, #The frame that was predicted
            "time_stamp":0, #time.time() value indicating when this prediction happend
            "frame_shape": [0,0],
            "speed_results": {
                "preprocess": None,
                "inference": None,
                "postprocess": None
            },
            "predictions":[
                {                         
                    "class_index":0,
                    "class_name":"NoName",
                    "bbox_confidence":0,
                    "bbox": [0,0,0,0], # Bounding box in the format [x1,y1,x2,y2]
                    "bbox_pixel_area": 0,
                    "is_coordinated_wrt_camera": False, # True if the coordinates are wrt the camera, False if they are wrt the frame
                    "belly_coordinate_wrt_camera": [0,0,0], # [x,y,z] coordinates of the object wrt the camera
                    "belly_distance_wrt_camera": 0, # distance between the camera and the object in meters
                    "keypoints": { # Keypoints are in the format [x,y,confidence]
                        "left_eye": [0,0,0],
                        "right_eye": [0,0,0],
                        "nose": [0,0,0],
                        "left_ear": [0,0,0],
                        "right_ear": [0,0,0],
                        "left_shoulder": [0,0,0],
                        "right_shoulder": [0,0,0],
                        "left_elbow": [0,0,0],
                        "right_elbow": [0,0,0],
                        "left_wrist": [0,0,0],
                        "right_wrist": [0,0,0],
                        "left_hip": [0,0,0],
                        "right_hip": [0,0,0],
                        "left_knee": [0,0,0],
                        "right_knee": [0,0,0],
                        "left_ankle": [0,0,0],
                        "right_ankle": [0,0,0]
                    }
                }
            ]
        }

    def get_model_class_names(self):
        """
        returns all class names that can be detected as a dict where the keys are integers and corresponds to the class indexes
        """

        return self.yolo_object.names
    
    def predict_frame(self, frame, h_angle = 105.5, v_angle = 57.5):
        """
        predicts the pose of a single frame and returns the results in the format specified in self.prediction_results. Also store the results in the class object.
        """
        results = self.yolo_object(frame, task = "pose")[0]
        
        self.prediction_results['frame'] = frame
        self.prediction_results['frame_shape'] = list(results.orig_shape) # Shape of the original image->  [height , width]
        self.prediction_results["speed_results"] = results.speed # {'preprocess': None, 'inference': None, 'postprocess': None}
        self.prediction_results["time_stamp"] = time.time()
       
        self.prediction_results["predictions"] = []
        for i, result in enumerate(results):
            #result corresponds to one of the predicted boxes. 
            boxes = result.boxes  # Boxes object for bbox outputs 

            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxy = boxes.xyxy.cpu().numpy()[0]
            bbox_pixel_area = (box_xyxy[2]-box_xyxy[0])*(box_xyxy[3]-box_xyxy[1])

            result_detection_dict = {                
                "class_index":box_cls_no,
                "class_name":box_cls_name,
                "bbox_confidence":box_conf,
                "bbox": box_xyxy, # Bounding box in the format [x1,y1,x2,y2]
                "bbox_pixel_area": bbox_pixel_area,
                "is_coordinated_wrt_camera": False,
                "belly_coordinate_wrt_camera": [0,0,0], # [x,y,z] coordinates of the object wrt the camera
                "belly_distance_wrt_camera": 0, # distance between the camera and the object in meters
                "keypoints": { # Keypoints are in the format [x,y,confidence,x_angle, y_angle]
                        "left_eye": [0,0,0,0,0],
                        "right_eye": [0,0,0,0,0],
                        "nose": [0,0,0,0,0],
                        "left_ear": [0,0,0,0,0],
                        "right_ear": [0,0,0,0,0],
                        "left_shoulder": [0,0,0,0,0],
                        "right_shoulder": [0,0,0,0,0],
                        "left_elbow": [0,0,0,0,0],
                        "right_elbow": [0,0,0,0,0],
                        "left_wrist": [0,0,0,0,0],
                        "right_wrist": [0,0,0,0,0],
                        "left_hip": [0,0,0,0,0],
                        "right_hip": [0,0,0,0,0],
                        "left_knee": [0,0,0,0,0],
                        "right_knee": [0,0,0,0,0],
                        "left_ankle": [0,0,0,0,0],
                        "right_ankle": [0,0,0,0,0],
                    }
            }       

            key_points = result.keypoints  # Keypoints object for pose outputs
            keypoint_confs = key_points.conf.cpu().numpy()[0]
            keypoints_xy = key_points.xy.cpu().numpy()[0]
                       
            frame_height = self.prediction_results['frame_shape'][0]
            frame_width = self.prediction_results['frame_shape'][1]

            #KEYPOINT_NAMES = ["left_eye", "rigt_eye", "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow" ,"right_elbow","left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
            for keypoint_index, keypoint_name in enumerate(poseDetector.KEYPOINT_NAMES):
                keypoint_conf = keypoint_confs[keypoint_index] 
                keypoint_x = keypoints_xy[keypoint_index][0]
                keypoint_y = keypoints_xy[keypoint_index][1]
                x_angle = ((keypoint_x/frame_width)-0.5)*h_angle
                y_angle = (0.5-(keypoint_y/frame_height))*v_angle

                result_detection_dict["keypoints"][keypoint_name] = [keypoint_x, keypoint_y , keypoint_conf, x_angle, y_angle]

            self.prediction_results["predictions"].append(result_detection_dict)

        return self.prediction_results

    def draw_bounding_boxes(self, confidence_threshold = 0.25):
        """
        Draws the bounding boxes predicted related to the last frame. Esnure that 'predict_frame' has been called before this function.
        """
        frame = self.prediction_results["frame"]

        for result in self.prediction_results["predictions"]:
            class_name = self.prediction_results["predictions"][0]["class_name"]
            confidence = result["bbox_confidence"]
            belly_distance = result["belly_distance_wrt_camera"]

            color_map = lambda x: (0, int(255 * (x)), int(255 * (1-x)) ) #BGR
            color = color_map(confidence)

            if result["bbox_confidence"] > confidence_threshold:
                cv2.rectangle(frame, (int(result["bbox"][0]), int(result["bbox"][1])), (int(result["bbox"][2]), int(result["bbox"][3])), color, 2)
                #TODO: ensure that distance is calculated
                if result["is_coordinated_wrt_camera"]:
                    cv2.putText(frame, f"{class_name}: {belly_distance:.2f}m ", (int(result["bbox"][0]), int(result["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"{class_name}", (int(result["bbox"][0]), int(result["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    def draw_keypoints_points(self, confidence_threshold = 0.25, DOT_SCALE_FACTOR = 1):
        """
        Draws the keypoints predicted related to the last frame. Esnure that 'predict_frame' has been called before this function.
        """
        DOT_MULTIPLIER = 0.00005
        frame = self.prediction_results["frame"]

        for result in self.prediction_results["predictions"]:

            dot_radius = math.ceil(DOT_SCALE_FACTOR*(DOT_MULTIPLIER*result["bbox_pixel_area"]))

            for keypoint_name in poseDetector.KEYPOINT_NAMES:
                keypoint = result["keypoints"][keypoint_name]
                if keypoint[2] > confidence_threshold:
                    # Set the radius for the border (stroke)
                    border_radius = dot_radius + 1  # Increase the radius slightly for the border

                    # Draw the black border
                    cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), border_radius, (0, 0, 0), -1)

                    # Draw the white filled circle
                    cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), dot_radius, (255, 255, 255), -1)

    def draw_upper_body_lines(self, confidence_threshold = 0.1):
        """
        Draws the upper body lines related to the last frame. Esnure that 'predict_frame' has been called before this function.
        """
        frame = self.prediction_results["frame"]

        joints_to_be_connected = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]
        for result in self.prediction_results["predictions"]:
            for keypoint_name in poseDetector.KEYPOINT_NAMES:
                if keypoint_name not in joints_to_be_connected:
                    continue

                keypoint = result["keypoints"][keypoint_name]

                for joint_name in joints_to_be_connected:
                    joint = result["keypoints"][joint_name]

                    check_1 = keypoint[0] > 0 and keypoint[1] > 0
                    check_2 = joint[0] > 0 and joint[1] > 0

                    if check_1 and check_2:
                        cv2.line(frame, (int(keypoint[0]), int(keypoint[1])), (int(joint[0]), int(joint[1])), (255, 255, 255), 2)

    def add_grid(self, row_count = 10, column_count = 10):
        """
        Draws a grid on the image
        """
        frame = self.prediction_results["frame"]
        frame_height = self.prediction_results['frame_shape'][0]
        frame_width = self.prediction_results['frame_shape'][1]

        for i in range(row_count):
            y = int(i * frame_height / row_count)
            stroke_width = 3 if i%(row_count/2)==0 else 1
            cv2.line(frame, (0, y), (frame_width, y), (0, 0, 0), stroke_width)


        for i in range(column_count):
            stroke_width = 3 if i%(row_count/2)==0 else 1
            x = int(i * frame_width / column_count)
            cv2.line(frame, (x, 0), (x, frame_height), (0, 0, 0), stroke_width)

    def draw_all(self):
        """
        Draws the bounding boxes, keypoints and upper body lines related to the last frame. Esnure that 'predict_frame' has been called before this function.
        """
        self.draw_bounding_boxes()
        self.draw_keypoints_points()
        self.draw_upper_body_lines()
        self.add_grid()

    def approximate_prediction_distance(self, h_view_angle = 105.5, v_view_angle = 57.5):
        """
        Calculates the distances between the camera and each detected person. if shoulders and hips are detected
        """
        for result in self.prediction_results["predictions"]:
            # Get the bounding box coordinates

            rs_data = result["keypoints"]["right_shoulder"]
            ls_data = result["keypoints"]["left_shoulder"]
            rh_data = result["keypoints"]["right_hip"]
            lh_data = result["keypoints"]["left_hip"]

            f_get_unit_vector = lambda angle_x, angle_y: [math.cos(math.radians(angle_y))*math.sin(math.radians(angle_x)), math.sin(math.radians(angle_y)), math.cos(math.radians(angle_y))* math.cos(math.radians(angle_x))]            
         
            rs_uv = f_get_unit_vector(rs_data[3], rs_data[4])
            ls_uv = f_get_unit_vector(ls_data[3], ls_data[4])
            rh_uv = f_get_unit_vector(rh_data[3], rh_data[4])
            lh_uv = f_get_unit_vector(lh_data[3], lh_data[4])

            unknowns = [0,0,0,0]
            unit_vectors = [rs_uv, ls_uv, rh_uv, lh_uv]
            def minimizer_function(unknowns, rs_uv, ls_uv, rh_uv, lh_uv)-> float:
                k_rs,k_ls,k_rh, k_lh = unknowns
                u_rs, u_ls, u_rh, u_lh = unit_vectors

                f_scale_vector = lambda vector, scale: [vector[0]*scale, vector[1]*scale, vector[2]*scale]

                pass



            continue
            
        

    

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")

    # model_path = input("Enter the path to the model: ")
    model_path = "C:\\Users\\Levovo20x\\Documents\\GitHub\\PPE-detection\\scripts\\object_detection\\models\\secret_yolov8n-pose.pt"

    detector = poseDetector(model_path)

    frame = cv2.imread(image_path) #1280, 1024

    detector.predict_frame(frame)
    detector.approximate_prediction_distance(h_view_angle= 128, v_view_angle= 102)    

    detector.draw_bounding_boxes()
    detector.draw_keypoints_points(DOT_SCALE_FACTOR = 0.5)
    detector.draw_upper_body_lines()
    
   
    cv2.imshow("frame", frame)
    cv2.waitKey(0)





