from ultralytics import YOLO
import cv2,math,time,os
import time

from scipy.optimize import minimize
import numpy as np

class poseDetector(): 
    KEYPOINT_NAMES = ["left_eye", "rigt_eye", "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow" ,"right_elbow","left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    
    #approximate distances between the keypoints of a person in meters
    SHOULDER_TO_SHOULDER = 0.36 
    SHOULDER_TO_HIP = 0.48 
    SHOULDER_TO_COUNTER_HIP = 0.53  
    SHOULDER_TO_ELBOW = 0.26

    def __init__(self, model_path : str ) -> None:
        # Check if the file exists
        if os.path.exists(model_path):
        # Check if the file has the specified extension
            _, file_extension = os.path.splitext(model_path)
            if file_extension != ".pt":
                raise ValueError(f"The model file should have the extension '.pt'.")
        else:
            raise FileNotFoundError(f"The model file does not exist.")

        self.MODEL_PATH = model_path        
        self.yolo_object = YOLO( self.MODEL_PATH, verbose= False)        
        self.prediction_results = None
        self._clear_prediction_results()
    
    def _clear_prediction_results(self):
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
                # The format of the prediction dictionary is specified in self._PREDICTION_DICT_TEMPLATE()
            ]
        }

    def _PREDICTION_DICT_TEMPLATE(self):
        empty_prediction_dict ={                         
                    "class_index":0,
                    "class_name":"NoName",
                    "bbox_confidence":0,
                    "bbox": [0,0,0,0], # Bounding box in the format [x1,y1,x2,y2]
                    "bbox_pixel_area": 0,
                    "is_coordinated_wrt_camera": False, # True if the coordinates are wrt the camera, False if they are wrt the frame
                    "belly_coordinate_wrt_camera": np.array([[0],[0],[0]]), # [x,y,z] coordinates of the object wrt the camera
                    "belly_distance_wrt_camera": 0, # distance between the camera and the object in meters
                    "is_coordinated_wrt_world_frame": False,
                    "belly_coordinate_wrt_world_frame":np.array([[0],[0],[0]]),
                    "belly_distance_wrt_world_frame": 0, # distance between the camera and the person's belly in meters
                    "is_pose_classified": False, # True if the pose is classified, False if not
                    "pose_classification": "NoPose", # The pose classification
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
        return empty_prediction_dict
    
    def predict_frame(self, frame, h_angle = 105.5, v_angle = 57.5):
        """
        predicts the pose of a single frame and returns the results in the format specified in self.prediction_results. Also store the results in the class object.
        """
        self._clear_prediction_results()

        results = self.yolo_object(frame, task = "pose", verbose= False)[0]        
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

            result_detection_dict = self._PREDICTION_DICT_TEMPLATE()
            result_detection_dict ["class_index"] = box_cls_no
            result_detection_dict ["class_name"] = box_cls_name
            result_detection_dict ["bbox_confidence"] = box_conf
            result_detection_dict ["bbox"] = box_xyxy # Bounding box in the format [x1,y1,x2,y2]
            result_detection_dict ["bbox_pixel_area"] = bbox_pixel_area

            key_points = result.keypoints  # Keypoints object for pose outputs
            keypoint_confs = key_points.conf.cpu().numpy()[0]
            keypoints_xy = key_points.xy.cpu().numpy()[0]
                       
            frame_height = self.prediction_results['frame_shape'][0]
            frame_width = self.prediction_results['frame_shape'][1]

            for keypoint_index, keypoint_name in enumerate(poseDetector.KEYPOINT_NAMES):
                keypoint_conf = keypoint_confs[keypoint_index] 
                keypoint_x = keypoints_xy[keypoint_index][0]
                keypoint_y = keypoints_xy[keypoint_index][1]
                if keypoint_x == 0 and keypoint_y == 0: #if the keypoint is not detected
                    #But this is also a prediction. Thus the confidence should not be set to zero. negative values are used to indicate that the keypoint is not detected
                    keypoint_conf = -keypoint_conf

                x_angle = ((keypoint_x/frame_width)-0.5)*h_angle
                y_angle = (0.5-(keypoint_y/frame_height))*v_angle

                result_detection_dict["keypoints"][keypoint_name] = [keypoint_x, keypoint_y , keypoint_conf, x_angle, y_angle]

            self.prediction_results["predictions"].append(result_detection_dict)

    def approximate_prediction_distance(self, box_condifence_threshold = 0.25, distance_threshold = 1, transformation_matrices = None):
        """
        Calculates the distances between the camera and each detected person. if shoulders and hips are detected

        box_condifence_threshold: minimum confidence of the bounding box to be considered while calculating distance
        distance_threshold: minimum distance that the belly of the person should be away from the camera to be considered while calculating distance in meters
        """
        for result in self.prediction_results["predictions"]:
            # Get the bounding box coordinates
            box_confidence = result["bbox_confidence"]
            if box_confidence < box_condifence_threshold:
                continue

            rs_data = result["keypoints"]["right_shoulder"]
            ls_data = result["keypoints"]["left_shoulder"]
            rh_data = result["keypoints"]["right_hip"]
            lh_data = result["keypoints"]["left_hip"]

            f_get_unit_vector = lambda angle_x, angle_y: [math.cos(math.radians(angle_y))*math.sin(math.radians(angle_x)), math.sin(math.radians(angle_y)), math.cos(math.radians(angle_y))* math.cos(math.radians(angle_x))]            
         
            rs_uv = f_get_unit_vector(rs_data[3], rs_data[4])
            ls_uv = f_get_unit_vector(ls_data[3], ls_data[4])
            rh_uv = f_get_unit_vector(rh_data[3], rh_data[4])
            lh_uv = f_get_unit_vector(lh_data[3], lh_data[4])

       
            def minimizer_function(unknowns, unit_vectors)-> float:
                k_rs,k_ls,k_rh, k_lh = unknowns
                u_rs, u_ls, u_rh, u_lh = unit_vectors

                f_scale_vector = lambda vector, scale: [vector[0]*scale, vector[1]*scale, vector[2]*scale]
                v_rs = f_scale_vector(u_rs, k_rs)
                v_ls = f_scale_vector(u_ls, k_ls)
                v_rh = f_scale_vector(u_rh, k_rh)
                v_lh = f_scale_vector(u_lh, k_lh)


                f_distance_between_vectors = lambda v1, v2: math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2)
                d_rs_ls = f_distance_between_vectors(v_rs, v_ls)
                d_rs_lh = f_distance_between_vectors(v_rs, v_lh)
                d_rs_rh = f_distance_between_vectors(v_rs, v_rh)
                d_ls_lh = f_distance_between_vectors(v_ls, v_lh)
                d_ls_rh = f_distance_between_vectors(v_ls, v_rh)
                
                #rs,ls,rh triangle error
                error_1 =  (d_rs_ls - poseDetector.SHOULDER_TO_SHOULDER)**2 + (d_rs_rh - poseDetector.SHOULDER_TO_HIP)**2 + (d_ls_rh - poseDetector.SHOULDER_TO_COUNTER_HIP)**2
                #rs,ls,lh triangle error
                error_2 = (d_rs_ls - poseDetector.SHOULDER_TO_SHOULDER)**2 + (d_ls_lh - poseDetector.SHOULDER_TO_HIP)**2 + (d_rs_lh - poseDetector.SHOULDER_TO_COUNTER_HIP)**2

                return (error_1 + error_2)
            
            #optimize the triangle
            tolerance = 1e-6
            bounds = [(0, 25), (0, 25), (0, 25), (0, 25)] #
            initial_guess = [5,5,5,5] 
            unit_vectors = [rs_uv, ls_uv, rh_uv, lh_uv]
            #NOTE: never remove comma after unit_vectors. Othewise it will be interpreted as a tuple
            minimizer_result = minimize(minimizer_function, initial_guess, args=( unit_vectors, ), method='L-BFGS-B', tol=tolerance, bounds = bounds)
            
            if minimizer_result.success == True:

                # k_rs, k_ls, k_rh, k_lh = unknowns
                scalars = minimizer_result.x
                v_rs = [rs_uv[0]*scalars[0], rs_uv[1]*scalars[0], rs_uv[2]*scalars[0]]
                v_ls = [ls_uv[0]*scalars[1], ls_uv[1]*scalars[1], ls_uv[2]*scalars[1]]
                v_rh = [rh_uv[0]*scalars[2], rh_uv[1]*scalars[2], rh_uv[2]*scalars[2]]
                v_lh = [lh_uv[0]*scalars[3], lh_uv[1]*scalars[3], lh_uv[2]*scalars[3]]

                v_belly = [(v_rs[0]+v_ls[0]+v_rh[0]+v_lh[0])/4, (v_rs[1]+v_ls[1]+v_rh[1]+v_lh[1])/4, (v_rs[2]+v_ls[2]+v_rh[2]+v_lh[2])/4]
                d_belly = math.sqrt(v_belly[0]**2 + v_belly[1]**2 + v_belly[2]**2)

                #v_belly = A(world_coordinate) + C
                A = transformation_matrices[0]
                C = transformation_matrices[1]

                v_belly = np.array(v_belly)
                v_belly = np.reshape(v_belly, (3,1))


                if d_belly > distance_threshold:
                    result["belly_coordinate_wrt_camera"] = v_belly
                    result["belly_distance_wrt_camera"] = d_belly
                    result["is_coordinated_wrt_camera"] = True       

                    world_coordinate = np.linalg.pinv(A) @ (v_belly - C)
                    result["belly_coordinate_wrt_world_frame"] = world_coordinate
                    result["belly_distance_wrt_world_frame"] = math.sqrt(world_coordinate[0][0]**2 + world_coordinate[1][0]**2 + world_coordinate[2][0]**2)
                    result["is_coordinated_wrt_world_frame"] = True

    def get_prediction_results(self):
        """
        returns the prediction results in the format specified in self.prediction_results
        """
        return self.prediction_results

    def get_person_detected_frame_regions(self):
        """
        returns the regions of a frame where a person was detected
        """        
        frame = self.prediction_results["frame"]
        person_regions = []
        for result in self.prediction_results["predictions"]:
            x1,y1,x2,y2 = result["bbox"]
            x1,y1,x2,y2 = int(x1), int(y1),int(x2),int(y2)

            # Extract the region of the frame where the person is detected
            person_region = frame[y1:y2, x1:x2]
            person_regions.append(person_region)

        return person_regions

    def get_max_confidence_among_results(self):
        """
        returns the maximum confidence among the predictions
        """
        max_confidence = 0
        for result in self.prediction_results["predictions"]:
            if result["bbox_confidence"] > max_confidence:
                max_confidence = result["bbox_confidence"]
        return max_confidence
    
    def draw_bounding_boxes(self, confidence_threshold = 0.25, add_blur = True, blur_kernel_size = 35):
        """
        Draws the bounding boxes predicted related to the last frame. Esnure that 'predict_frame' has been called before this function.
        """
        frame = self.prediction_results["frame"]

        for result in self.prediction_results["predictions"]:
            class_name = self.prediction_results["predictions"][0]["class_name"]
            confidence = result["bbox_confidence"]
            belly_distance = result["belly_distance_wrt_camera"]
            belly_vector = result["belly_coordinate_wrt_world_frame"]

            color_map = lambda x: (0, int(255 * (x)), int(255 * (1-x)) ) #BGR
            color = color_map(confidence)

            if result["bbox_confidence"] > confidence_threshold:
                x1, y1, x2, y2 = result["bbox"]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                if add_blur:
                    roi = frame[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (blur_kernel_size, blur_kernel_size), 0)
                    frame[y1:y2, x1:x2] = blurred_roi

                if result["is_coordinated_wrt_world_frame"]:
                    cv2.putText(frame, f"{belly_distance:.1f}m : ({belly_vector[0][0]:.1f}, {belly_vector[1][0]:.1f}, {belly_vector[2][0]:.1f})", (int(result["bbox"][0]), int(result["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"{class_name}", (int(result["bbox"][0]), int(result["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

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

    def draw_grid(self, row_count = 10, column_count = 10):
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
        self.draw_grid()

    def get_world_coordinates(self):
        """
        returns the world coordinates of the belly of the detected persons
        """
        world_coordinates = []
        for result in self.prediction_results["predictions"]:
            world_coordinates.append(result["belly_coordinate_wrt_world_frame"])
        return world_coordinates
    



