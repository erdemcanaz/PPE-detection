import json, pprint

class MemorylessViolationEvaluator:
    def __init__(self):                
        with open('json_files/rules_and_evaluator_configs.json') as f:
            rules = json.load(f)

        self.DETECT_HARD_HAT_VIOLATION = rules["violation_evaluator_config"]["detect_hard_hat_violation"]
        self.DETECT_RESTRICTED_AREA_VIOLATION = rules["violation_evaluator_config"]["detect_restricted_area_violation"]
        
        self.HEIGHT_RULE_THRESHOLD = rules["violation_evaluator_config"]["height_rule_threshold_height"]
        self.POSE_CONFIDENCE_THRESHOLD = rules["violation_evaluator_config"]["confidence_thresholds"]["pose_confidence_threshold"]
        self.FORKLIFT_CONFIDENCE_THRESHOLD = rules["violation_evaluator_config"]["confidence_thresholds"]["forklift_confidence_threshold"]
        self.HARD_HAT_CONFIDENCE_THRESHOLD = rules["violation_evaluator_config"]["confidence_thresholds"]["hard_hat_confidence_threshold"]
        self.FORKLIFT_PERSON_OVERLAPPING_THRESHOLD = rules["violation_evaluator_config"]["confidence_thresholds"]["forklift_person_overlapping_threshold"]
        self.HARDHAT_PERSON_OVERLAPPING_THRESHOLD = rules["violation_evaluator_config"]["confidence_thresholds"]["hardhat_person_overlapping_threshold"]
        
        self.HUMAN_RESTRICTED_VOLUMES= rules["rule_areas"]["human_restricted_area_rule_applied_volumes"]
        self.HARD_HAT_RULE_VOLUMES = rules["rule_areas"]["hard_hat_rule_applied_volumes"]
        
    def evaluate_for_violations(self, detections:dict[list]):
        detected_forklifts_as_objects = []
        detected_persons_as_objects = []
        detected_hard_hats_as_objects = []

        #for detections "bbox_confidence" is higher than the threshold, initilize detections ojects. Pose detections are also checked for "is_coordinated_wrt_world_frame" attribute
        for forklift_detection_dict in detections["forklift_detections"]: 
            if forklift_detection_dict["bbox_confidence"] >= self.FORKLIFT_CONFIDENCE_THRESHOLD:
                detected_forklifts_as_objects.append(ForkliftDetection(detection_dict=forklift_detection_dict))
        for hard_hat_detection_dict in detections["hard_hat_detections"]:
            if hard_hat_detection_dict["bbox_confidence"] >= self.HARD_HAT_CONFIDENCE_THRESHOLD:
                detected_hard_hats_as_objects.append(HardHatDetection(detection_dict=hard_hat_detection_dict))
        for pose_detection_dict in detections["pose_detections"]:
            if pose_detection_dict["bbox_confidence"] >= self.POSE_CONFIDENCE_THRESHOLD and pose_detection_dict["is_coordinated_wrt_world_frame"]:
                detected_persons_as_objects.append(PersonDetection(detection_dict=pose_detection_dict))
       
        #match forklifts with humans if possible.
        for forklift_detection_obj in detected_forklifts_as_objects:
            best_person_candidate = [None, 0] #person detection object, overlapping ratio

            for person_detection_obj in detected_persons_as_objects:
                forklift_and_person_overlapping_ratio = Detection.return_overlapping_bbox_ratio_with(forklift_detection_obj, person_detection_obj)

                is_person_already_matched = person_detection_obj.is_matched_with_forklift()
                is_overlap_ratio_higher_than_threshold = forklift_and_person_overlapping_ratio > self.FORKLIFT_PERSON_OVERLAPPING_THRESHOLD
                is_no_candidate_yet = best_person_candidate[0] is None
                is_better_overlap_ratio = forklift_and_person_overlapping_ratio > best_person_candidate[1]
                is_same_overlap_ratio = forklift_and_person_overlapping_ratio == best_person_candidate[1]
                is_higher_person_confidence = person_detection_obj.get_bbox_confidence() > best_person_candidate[0].get_bbox_confidence() if best_person_candidate[0] is not None else True

                condition_1 = (not is_person_already_matched) and is_overlap_ratio_higher_than_threshold and (is_no_candidate_yet or is_better_overlap_ratio)
                condition_2 = (not is_person_already_matched) and is_overlap_ratio_higher_than_threshold and (is_same_overlap_ratio and is_higher_person_confidence)

                if condition_1 or condition_2:
                    best_person_candidate[0], best_person_candidate[1] = person_detection_obj, forklift_and_person_overlapping_ratio
             
            if best_person_candidate[0] is not None:
                forklift_detection_obj.match_with_person(best_person_candidate[0])
                best_person_candidate[0].match_with_forklift(forklift_detection_obj)
                print("Matched forklift with person")

        #match person with hard hat if possible
        for person_detection_obj in detected_persons_as_objects:
            best_hard_hat_candidate = [None, 0, float('inf')] #hard hat detection object; overlapping ratio; distance between head center and hardhat centers

            for hard_hat_detection_obj in detected_hard_hats_as_objects:
                hard_hat_and_person_overlapping_ratio = Detection.return_overlapping_bbox_ratio_with(hard_hat_detection_obj, person_detection_obj)
                distance_between_centers = Detection.return_pixel_distance_between_centers(hard_hat_detection_obj.get_bbox_center_px(), person_detection_obj.get_head_center_px())

                is_hard_hat_already_matched = hard_hat_detection_obj.is_matched_with_person()
                is_overlap_ratio_higher_than_threshold = hard_hat_and_person_overlapping_ratio > self.HARDHAT_PERSON_OVERLAPPING_THRESHOLD
                is_no_candidate_yet = best_hard_hat_candidate[0] is None
                is_better_overlap_ratio = hard_hat_and_person_overlapping_ratio > best_hard_hat_candidate[1]                
                is_same_overlap_ratio = hard_hat_and_person_overlapping_ratio == best_hard_hat_candidate[1]
                is_closer_to_head_center = distance_between_centers < best_hard_hat_candidate[2]

                print(f"hard_hat_and_person_overlapping_ratio: {hard_hat_and_person_overlapping_ratio}, distance_between_centers: {distance_between_centers}")
                pprint.pprint(f"is_hard_hat_already_matched: {is_hard_hat_already_matched}, is_overlap_ratio_higher_than_threshold: {is_overlap_ratio_higher_than_threshold}, is_no_candidate_yet: {is_no_candidate_yet}, is_better_overlap_ratio: {is_better_overlap_ratio}, is_same_overlap_ratio: {is_same_overlap_ratio}, is_closer_to_head_center: {is_closer_to_head_center}")
                condition_1 = (not is_hard_hat_already_matched) and is_overlap_ratio_higher_than_threshold and (is_no_candidate_yet or is_better_overlap_ratio)
                condition_2 = (not is_hard_hat_already_matched) and is_overlap_ratio_higher_than_threshold and (is_same_overlap_ratio and is_closer_to_head_center)
                print(condition_1, condition_2)
                if condition_1 or condition_2:
                    best_hard_hat_candidate[0], best_hard_hat_candidate[1], best_hard_hat_candidate[2] = hard_hat_detection_obj, hard_hat_and_person_overlapping_ratio, distance_between_centers
            
            if best_hard_hat_candidate[0] is not None:
                person_detection_obj.match_with_hard_hat(best_hard_hat_candidate[0])
                best_hard_hat_candidate[0].match_with_person(person_detection_obj)
                print("Matched person with hard hat")

        #evaluate for violations
        # -> check if human is inside a strictly restricted area
        # -> check if human is inside a restricted area (if in forklift, not violating)
        # -> check if human restricing the hard hat rule
        #     -> if not wearing hard hat, check if human is inside a hard hat rule area
        # -> check if human is above the height threshold  

    
        #return human objects
                
class Detection:
    def __init__(self, detection_dict:dict = None):
        self.frame_shape = detection_dict["frame_shape"] # (height, width)
        self.class_name = detection_dict["class_name"]
        self.bbox_confidence = detection_dict["bbox_confidence"]
        self.bbox_xyxy_px = detection_dict["bbox_xyxy_px"]
        self.bbox_center_px = detection_dict["bbox_center_px"]

    def get_bbox_area(self):
        return (self.bbox_xyxy_px[2] - self.bbox_xyxy_px[0]) * (self.bbox_xyxy_px[3] - self.bbox_xyxy_px[1])
    
    def get_bbox_center_px(self):
        return self.bbox_center_px
    
    def get_bbox_confidence(self):
        return self.bbox_confidence
    
    def return_overlapping_bbox_ratio_with(first_detection = None, other_detection = None) -> float:
        normalizer_area = min(first_detection.get_bbox_area(), other_detection.get_bbox_area())
        
        intersection_upper_left_x = max(first_detection.bbox_xyxy_px[0], other_detection.bbox_xyxy_px[0])
        intersection_lower_right_x = min(first_detection.bbox_xyxy_px[2], other_detection.bbox_xyxy_px[2])
        intersection_upper_left_y = max(first_detection.bbox_xyxy_px[1], other_detection.bbox_xyxy_px[1])
        intersection_lower_right_y = min(first_detection.bbox_xyxy_px[3], other_detection.bbox_xyxy_px[3])

        intersection_width = intersection_lower_right_x - intersection_upper_left_x 
        intersection_height = intersection_lower_right_y - intersection_upper_left_y       
        
        if intersection_width > 0 and intersection_height > 0:
            intersection_area = intersection_width * intersection_height
            return intersection_area / normalizer_area
        else:
            return 0 #no intersection since width or height is negative 

    def return_pixel_distance_between_centers(center_1:list = None,  center_2:list = None):
        return ((center_1[0] - center_2[0])**2 + (center_1[1] - center_2[1])**2)**0.5 #Note actualy there is no need to take square root in this concept, but it is more readable this way.
        
   
class HardHatDetection(Detection):
    def __init__(self, detection_dict:dict = None):
        super().__init__(detection_dict)        
        self.matched_person_detection_obj = None

    def match_with_person(self, person_detection_obj):
        self.matched_person_detection_obj = person_detection_obj

    def is_matched_with_person(self)->bool:
        return self.matched_person_detection_obj != None
    
class ForkliftDetection(Detection):
    def __init__(self, detection_dict:dict = None):
        super().__init__(detection_dict)
        self.matched_person_detection_obj = None

    def match_with_person(self, person_detection_obj):
        self.matched_person_detection_obj = person_detection_obj

    def is_matched_with_person(self)->bool:
        return self.matched_person_detection_obj != None

class PersonDetection(Detection):
    def __init__(self, detection_dict:dict = None):
        super().__init__(detection_dict)

        self.matched_forklift_det_obj = None
        self.matched_hard_hat_det_obj = None

        self.is_coordianted_wrt_world_frame = detection_dict["is_coordinated_wrt_world_frame"]
        self.belly_coordinate_wrt_world_frame = detection_dict["belly_coordinate_wrt_world_frame"]
        self.keypoints = detection_dict["keypoints"] # Keypoints are in the format [x,y,confidence,x_angle, y_angle]

        self.head_center_px = self.calculate_head_center_px()

    def get_head_center_px(self):
        return self.head_center_px
    
    def is_matched_with_forklift(self)->bool:
        return self.matched_forklift_det_obj != None
    
    def match_with_forklift(self, forklift_detection_obj):
        self.matched_forklift_det_obj = forklift_detection_obj

    def is_matched_with_hard_hat(self)->bool:
        return self.matched_hard_hat_det_obj != None
    
    def match_with_hard_hat(self, hard_hat_detection_obj):
        self.matched_hard_hat_det_obj = hard_hat_detection_obj
        
    def calculate_head_center_px(self)->list:
        interested_keypoint_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
        result =[0, 0, 0] #sum px (head center x), sum py (head center y), count
        for keypoint_name in interested_keypoint_names:
            keypoint_x, keypoint_y, keypoint_conf, _, _ = self.keypoints[keypoint_name] # [x,y,confidence,x_angle, y_angle]
            if keypoint_conf > 0: #means keypoint is detected, negative confidence means not detected
                result[0] += keypoint_x
                result[1] += keypoint_y
                result[2] += 1

        if result[2] != 0:            
            result[0] = result[0] / result[2]
            result[1] = result[1] / result[2]
        return (result[0], result[1]) #TODO: 0,0 means not any of the head joint is detected. This is not an elegant solution (i.e bug), but it is not important for now.






