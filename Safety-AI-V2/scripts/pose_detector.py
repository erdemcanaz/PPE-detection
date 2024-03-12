from ultralytics import YOLO
import cv2,math,time,os
import time

from scripts.camera import Camera

from scipy.optimize import minimize
import numpy as np

class PoseDetector(): 
    #keypoints detected by the model in the detection order
    KEYPOINT_NAMES = ["left_eye", "rigt_eye", "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow" ,"right_elbow","left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    
    #approximate distances between the keypoints of a person in meters (1.75m)
    SHOULDER_TO_SHOULDER = 0.36 
    SHOULDER_TO_HIP = 0.48 
    SHOULDER_TO_COUNTER_HIP = 0.53  
    SHOULDER_TO_ELBOW = 0.26

    def __init__(self, model_path : str ) -> None:      
        self.MODEL_PATH = model_path        
        self.yolo_object = YOLO( self.MODEL_PATH, verbose= False)        
        self.recent_prediction_results = None # This will be a list of dictionaries, each dictionary will contain the prediction results for a single detection

    def get_empty_prediction_dict_template(self, camera_object:Camera=None) -> dict:
        empty_prediction_dict = {   
                    "DETECTOR_TYPE":"PoseDetector",                             # which detector made this prediction
                    "frame_shape": [0,0],                                       # [0,0], [height , width] in pixels
                    "class_name":"",                                            # hard_hat, no_hard_hat
                    "bbox_confidence":0,                                        # 0.0 to 1.0
                    "bbox_xyxy_px":[0,0,0,0],                                   # [x1,y1,x2,y2] in pixels
                    "bbox_center_px": [0,0],                                    # [x,y] in pixels

                    #------------------pose specific fields------------------
                    "is_coordinated_wrt_camera": False,                         # True if the coordinates are wrt the camera, False if they are wrt the frame
                    "belly_coordinate_wrt_camera": np.array([[0],[0],[0]]),     # [x,y,z] coordinates of the object wrt the camera
                    "is_coordinated_wrt_world_frame": False,
                    "belly_coordinate_wrt_world_frame":np.array([[0],[0],[0]]),
                    "keypoints": {                                              # Keypoints are in the format [x,y,confidence,x_angle, y_angle]
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
    
    def predict_frame_and_return_detections(self, frame, camera_object:Camera = None) -> list[dict]:
        self.recent_prediction_results = []
        
        results = self.yolo_object(frame, task = "pose", verbose= False)[0]
        for i, result in enumerate(results):
            boxes = result.boxes
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]
            if box_cls_name not in ["person"]:
                continue
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxy = boxes.xyxy.cpu().numpy()[0]

            prediction_dict_template = self.get_empty_prediction_dict_template()
            prediction_dict_template["frame_shape"] = list(results.orig_shape)
            prediction_dict_template["class_name"] = box_cls_name
            prediction_dict_template["bbox_confidence"] = box_conf
            prediction_dict_template["bbox_xyxy_px"] = box_xyxy # Bounding box in the format [x1,y1,x2,y2]
            prediction_dict_template["bbox_center_px"] = [ (box_xyxy[0]+box_xyxy[2])/2, (box_xyxy[1]+box_xyxy[3])/2]
            
            key_points = result.keypoints  # Keypoints object for pose outputs
            keypoint_confs = key_points.conf.cpu().numpy()[0]
            keypoints_xy = key_points.xy.cpu().numpy()[0]
                       
            frame_height = prediction_dict_template['frame_shape'][0]
            frame_width = prediction_dict_template['frame_shape'][1]
            h_angle, v_angle = camera_object.get_camera_view_angles()
            for keypoint_index, keypoint_name in enumerate(PoseDetector.KEYPOINT_NAMES):
                keypoint_conf = keypoint_confs[keypoint_index] 
                keypoint_x = keypoints_xy[keypoint_index][0]
                keypoint_y = keypoints_xy[keypoint_index][1]
                if keypoint_x == 0 and keypoint_y == 0: #if the keypoint is not detected
                    #But this is also a prediction. Thus the confidence should not be set to zero. negative values are used to indicate that the keypoint is not detected
                    keypoint_conf = -keypoint_conf

                x_angle = ((keypoint_x/frame_width)-0.5)*h_angle
                y_angle = (0.5-(keypoint_y/frame_height))*v_angle

                prediction_dict_template["keypoints"][keypoint_name] = [keypoint_x, keypoint_y , keypoint_conf, x_angle, y_angle]

            self.recent_prediction_results.append(prediction_dict_template)
        
        return self.recent_prediction_results
    
    def approximate_prediction_distance(self, prediction_dict:dict= None,  distance_threshold = 1, shoulders_confidence_threshold = 0.75, camera_object:Camera = None):
        """
        Calculates the distances between the camera and each detected person. if shoulders and hips are detected

        box_condifence_threshold: minimum confidence of the bounding box to be considered while calculating distance
        distance_threshold: minimum distance that the belly of the person should be away from the camera to be considered while calculating distance in meters
        """
        DISTANCE_THRESHOLD = 1 # Min distance in meters from camera to belly in meters
        SHOULDERS_CONFIDENCE_THRESHOLD = 0.75 # Min confidence of the shoulders to be considered while calculating distance
    
        for result in self.prediction_results["predictions"]:
            # Get the bounding box coordinates
          

            rs_data = result["keypoints"]["right_shoulder"]
            ls_data = result["keypoints"]["left_shoulder"]
            rh_data = result["keypoints"]["right_hip"]
            lh_data = result["keypoints"]["left_hip"]

            if rs_data[2] < shoulders_confidence_threshold or ls_data[2] < shoulders_confidence_threshold:
                #to calculate distance, it is necessary to have the two shoulder keypoints by this algorithm
                continue

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
                error_1 =  (d_rs_ls - PoseDetector.SHOULDER_TO_SHOULDER)**2 + (d_rs_rh - PoseDetector.SHOULDER_TO_HIP)**2 + (d_ls_rh - PoseDetector.SHOULDER_TO_COUNTER_HIP)**2
                #rs,ls,lh triangle error
                error_2 = (d_rs_ls - PoseDetector.SHOULDER_TO_SHOULDER)**2 + (d_ls_lh - PoseDetector.SHOULDER_TO_HIP)**2 + (d_rs_lh - PoseDetector.SHOULDER_TO_COUNTER_HIP)**2

                error = error_1+ error_2
                return error
            
            #optimize the triangle
            tolerance = 1e-6
            initial_guess = [5,5,5,5] 
            unit_vectors = [rs_uv, ls_uv, rh_uv, lh_uv]
            #NOTE: never remove comma after unit_vectors. Othewise it will be interpreted as a tuple
            minimizer_result = minimize(minimizer_function, initial_guess, args=( unit_vectors, ), method='L-BFGS-B', tol=tolerance)

            if minimizer_result.success == True:
                # k_rs, k_ls, k_rh, k_lh = unknowns
                scalars = minimizer_result.x
                v_rs = [rs_uv[0]*scalars[0], rs_uv[1]*scalars[0], rs_uv[2]*scalars[0]]
                v_ls = [ls_uv[0]*scalars[1], ls_uv[1]*scalars[1], ls_uv[2]*scalars[1]]
                v_rh = [rh_uv[0]*scalars[2], rh_uv[1]*scalars[2], rh_uv[2]*scalars[2]]
                v_lh = [lh_uv[0]*scalars[3], lh_uv[1]*scalars[3], lh_uv[2]*scalars[3]]

                
                if rh_data[2]>lh_data[2]:
                    #
                    v_belly = [ (v_ls[0]+v_rh[0])/2, (v_ls[1]+v_rh[1])/2, (v_ls[2]+v_rh[2])/2 ]
                    d_belly = math.sqrt(v_belly[0]**2 + v_belly[1]**2 + v_belly[2]**2)
                else:
                    v_belly = [ (v_rs[0]+v_lh[0])/2, (v_rs[1]+v_lh[1])/2, (v_rs[2]+v_lh[2])/2 ]
                    d_belly = math.sqrt(v_belly[0]**2 + v_belly[1]**2 + v_belly[2]**2)

                #v_belly = A(world_coordinate) + C
                A = transformation_matrices[0]
                C = transformation_matrices[1]
                T = transformation_matrices[2] if len(transformation_matrices) >= 3 else [0,0,0]


                v_belly = np.array(v_belly)
                v_belly = np.reshape(v_belly, (3,1))

                if d_belly > distance_threshold:
                    result["is_coordinated_wrt_camera"] = True       
                    result["belly_coordinate_wrt_camera"] = v_belly
                    result["belly_distance_wrt_camera"] = d_belly

                    world_coordinate = np.linalg.pinv(A) @ (v_belly - C)
                    world_coordinate[0][0]= world_coordinate[0][0] + T[0]
                    world_coordinate[1][0]= world_coordinate[1][0] + T[1]
                    world_coordinate[2][0]= world_coordinate[2][0] + T[2]
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
                    cv2.putText(frame, f"{belly_distance:.1f}m : ({belly_vector[0][0]:.2f}, {belly_vector[1][0]:.2f}, {belly_vector[2][0]:.2f})", (int(result["bbox"][0]), int(result["bbox"][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
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

            for keypoint_name in PoseDetector.KEYPOINT_NAMES:
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
            for keypoint_name in PoseDetector.KEYPOINT_NAMES:
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
    



