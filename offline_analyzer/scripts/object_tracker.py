import cv2
import random

class TrackerSupervisor:
    def __init__(self, max_age: int = 10, max_px_distance = 1000, confidence_threshold = 0.5, speed_attenuation_constant = 0.9)-> None:
        self.last_id = 0
        self.object_trackers = [] 
        self.MAX_AGE = max_age
        self.MAX_PX_DISTANCE = max_px_distance
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.SPEED_ATTENUATION_CONSTANT = speed_attenuation_constant

    def clear_trackers(self):
        for tracker in self.object_trackers:
            del tracker
        self.object_trackers = []
        print("All trackers are cleared")

    def update_trackers_with_detections(self, detections):
        #detections are the center coordinates of the bounding boxes, confidences and the x,y,z coordinates of the person
        
        matched_trackers = []
        matched_detections = []

        for detection in detections:
            #detection = {"bbox_center" = [px, py], "confidence" = 0.9, "person_coordinate"=[px, py, pz]}

            box_center_px, box_center_py = detection["bbox_center"]
            best_tracker_match = None
            best_distance = None

            for tracker in self.object_trackers:
                if tracker in matched_trackers:
                    continue                
               
                distance_now = tracker.calculate_distance(box_center_px, box_center_py)
                if distance_now < self.MAX_PX_DISTANCE:
                    if best_distance==None or best_distance> distance_now:
                        best_distance = distance_now 
                        best_tracker_match = tracker                      

            if best_tracker_match is not None:
                best_tracker_match.set_position(box_center_px, box_center_py)
                matched_trackers.append(best_tracker_match)          
                matched_detections.append(detection)  

        for tracker in self.object_trackers:
            if tracker not in matched_trackers:
                if tracker.is_old():
                    print(f"Tracker {tracker.TRACK_ID} is old and removed")
                    self.object_trackers.remove(tracker)

                tracker.update_position_using_speed(attenuation_factor = self.SPEED_ATTENUATION_CONSTANT)

        for detection in detections:
            if detection not in matched_detections:
                #some detections are on top of each other due to the nature of the pose detector. We ignore the ones with low confidence since they are likely to be the same person              
                #IF the confidence is low, pass this detection
                if detection["confidence"] < self.CONFIDENCE_THRESHOLD:
                    continue 
                #IF there is already a very close tracker (i.e. inside box), pass this detection
                pass_detection = False
                x1,y1,x2,y2 = detection["bbox_xyxy"]
                for tracker in self.object_trackers:
                    tx1,ty1 = tracker.get_last_position()
                    if (x1<tx1<x2) and (y1<ty1<y2):
                        pass_detection = True
                        break
                if pass_detection:
                    continue                  
                            
                self.last_id = self.last_id + 1
                self.object_trackers.append(Tracker(track_id = self.last_id, max_age = self.MAX_AGE, px = detection["bbox_center"][0], py = detection["bbox_center"][1]))

    def draw_trackers(self,frame):
        for tracker in self.object_trackers:
            tracker_name = f"ID: {tracker.TRACK_ID}"          
            center_px= int(tracker.get_last_position()[0])
            center_py= int(tracker.get_last_position()[1])

            color = (0, 255, 0)
            if tracker.get_track_state() == 0:
                color = (0, 0, 255)
            cv2.putText(frame, tracker_name, (center_px, center_py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (center_px, center_py), 4, color, -1)          

class Tracker:

    def __init__(self, track_id: int, max_age: int = None, px: int = None, py: int = None):        
        self.TRACK_ID = track_id
        self.MAX_AGE = max_age
        self.state = 1 # 1: tracking, 0: waiting
        self.current_position = [px, py]
        self.age = 0
        self.last_position = None
        self.speed = [0,0] # px/s in x and y direction respectively

    def get_track_id(self) -> int:
        return self.track_id
    
    def get_track_state(self) -> int:
        return self.state
    
    def set_position(self, px: int, py: int) -> None:
        print(f"Tracker {self.TRACK_ID} is updated with new position: {px}, {py}")
        self.state = 1
        self.age = 0
        self.last_position = self.current_position
        self.current_position = [px, py]
        self.speed[0] = self.current_position[0] - self.last_position[0]
        self.speed[1] = self.current_position[1] - self.last_position[1]

    def update_position_using_speed(self, attenuation_factor = 1) -> None:
        self.state = 0
        self.age +=1
        self.last_position = self.current_position
        self.current_position[0] += self.speed[0]*attenuation_factor
        self.current_position[1] += self.speed[1]*attenuation_factor
        print(f"Tracker {self.TRACK_ID} is updated with new position: {self.current_position[0]}, {self.current_position[1]} using speed data")

    def calculate_distance(self, px, py) -> float:
        return ((px - self.current_position[0])**2 + (py - self.current_position[1])**2)**0.5
    
    def is_old(self) -> bool:
        return self.age > self.MAX_AGE    

    def get_last_position(self):
        return self.current_position
    

