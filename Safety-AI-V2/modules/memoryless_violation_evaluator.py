import json

class MemorylessViolationEvaluator:
    def __init__(self):                
        with open('json_files/rules_and_evaluator_configs.json') as f:
            rules = json.load(f)

        self.DETECT_HARD_HAT_VIOLATION = rules["violation_evaluator_config"]["detect_hard_hat_violation"]
        self.DETECT_RESTRICTED_AREA_VIOLATION = rules["violation_evaluator_config"]["detect_restricted_area_violation"]
        
        self.HEIGHT_RULE_THRESHOLD = rules["violation_evaluator_config"]["height_rule_threshold_height"]
        self.POSE_CONFIDENCE_THRESHOLD = rules["violation_evaluator_config"]["confidence_thresholds"]["pose_confidence_threshold"]
        self.HARD_HAT_CONFIDENCE_THRESHOLD = rules["violation_evaluator_config"]["confidence_thresholds"]["hard_hat_confidence_threshold"]

        self.HUMAN_RESTRICTED_AREAS = rules["rule_areas"]["human_restricted_area_rule_applied_areas"]
        self.STRICT_RESTRICTED_AREAS = rules["rule_areas"]["strict_restricted_area_rule_applied_areas"]
        self.HARD_HAT_RULE_AREAS = rules["rule_areas"]["hard_hat_rule_applied_areas"]

    def evaluate_for_violations(self, detections:dict[list]):
        detected_forklifts = []
        detected_persons = []
        detected_hard_hats = []

        #match humans with forklifts if possible
        # -> check intersection of bboxs
        # -> if multiple perfect matches, choose the one with higher confidence
        # -> if multiple matches with same confidence, choose the one with smaller distance to the center of the forklift bbox

        #match hard hats with humans if possible
        # -> check if head keypoints are inside the bbox
        # -> check intersection of bboxs
        # -> if multiple perfect matches, choose human head center (mean of the keypoints) closest to the center of the hard hat bbox center

        #evaluate for violations
        # -> check if human is inside a strictly restricted area
        # -> check if human is inside a restricted area (if in forklift, not violating)
        # -> check if human restricing the hard hat rule
        #     -> if not wearing hard hat, check if human is inside a hard hat rule area
        # -> check if human is above the height threshold  

    
        #return human objects
        
class HardHatDetection:
    def __init__(self, bbox_confidence:float,):
        self.bbox_confidence = bbox_confidence
        self.matched_human_detection_det = None

class ForkliftDetection:
    def __init__(self, bbox_confidence:float):
        self.bbox_confidence = bbox_confidence
        self.matched_human_detection_det = None

class HumanDetection:
    def __init__(self, bbox_confidence:float, is_coordinated:bool, coordinates:list[int], inside_forklift:bool, wearing_hard_hat:bool, is_violating:bool, is_violating_restricted_area:bool, is_above_height_threshold:bool, is_violating_hard_hat_rule:bool):
        self.bbox_confidence = bbox_confidence

        self.is_coordinated = is_coordinated
        self.coordinates = coordinates

        self.matched_forklift_det = None
        self.matched_hard_hat_det = None

        self.wearing_hard_hat = wearing_hard_hat
        self.is_violating = is_violating
        self.is_violating_restricted_area = is_violating_restricted_area
        self.is_above_height_threshold = is_above_height_threshold
        self.is_violating_hard_hat_rule = is_violating_hard_hat_rule

        self.is_matched_with_forklift = False




