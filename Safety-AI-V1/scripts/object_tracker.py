import cv2
import random


class TrackerSupervisor:
    def __init__(self, max_age: int = 10, max_px_distance=1000, confidence_threshold=0.5) -> None:
        self.last_id = 0
        self.object_trackers = []

        self.tracker_records = {}

        self.MAX_AGE = max_age
        self.MAX_PX_DISTANCE = max_px_distance
        self.CONFIDENCE_THRESHOLD = confidence_threshold

    def clear_trackers(self):
        self.object_trackers = []
        self.tracker_records = {}
        print("All trackers are cleared")

    def get_tracker_records(self) -> list:
        for tracker in self.object_trackers:
            self.tracker_records[str(tracker.get_track_id())
                                 ] = tracker.get_tracker_record()

        return self.tracker_records

    def update_trackers_with_detections(self, detections: list[dict] = None):
        # detections are the center coordinates of the bounding boxes, confidences and the x,y,z coordinates of the person

        matched_trackers = []
        matched_detections = []

        for detection in detections:
            detection["px"] = (detection["bbox_coordinates"][0] +
                               detection["bbox_coordinates"][2])/2  # px-center of the box
            detection["py"] = (detection["bbox_coordinates"][1] +
                               detection["bbox_coordinates"][3])/2  # py-center of the box

            best_tracker_match = None
            best_distance = None

            for tracker in self.object_trackers:
                if tracker in matched_trackers:
                    continue

                distance_now = tracker.calculate_distance(
                    detection["px"], detection["py"])
                if distance_now < self.MAX_PX_DISTANCE:
                    if best_distance == None or best_distance > distance_now:
                        best_distance = distance_now
                        best_tracker_match = tracker

            if best_tracker_match is not None:
                best_tracker_match.set_position(track_current_info=detection)
                matched_trackers.append(best_tracker_match)
                matched_detections.append(detection)

        for tracker in self.object_trackers:
            if tracker not in matched_trackers:
                if tracker.is_old():
                    self.tracker_records[str(
                        tracker.get_track_id())] = tracker.get_tracker_record()
                    print(f"Tracker {tracker.TRACK_ID} is old and removed")
                    self.object_trackers.remove(tracker)
                else:
                    # @TODO: guess the position of the tracker using speed data
                    pass

        for detection in detections:
            if detection not in matched_detections:

                # some detections are on top of each other due to the nature of the pose detector. We ignore the ones with low confidence since they are likely to be the same person
                # IF the confidence is low, pass this detection
                if float(detection["bbox_confidence"]) < self.CONFIDENCE_THRESHOLD:
                    continue
                # IF there is already a very close tracker (i.e. inside box), pass this detection
                pass_detection = False
                x1, y1, x2, y2 = detection["bbox_coordinates"]
                for tracker in self.object_trackers:
                    tx1, ty1 = tracker.get_last_position()
                    if (x1 < tx1 < x2) and (y1 < ty1 < y2):
                        pass_detection = True
                        break
                if pass_detection:
                    continue

                self.last_id = self.last_id + 1
                self.object_trackers.append(Tracker(
                    track_id=self.last_id, max_age=self.MAX_AGE, track_current_info=detection))

    def draw_trackers(self, frame):
        for tracker in self.object_trackers:
            tracker_name = f"ID: {tracker.get_track_id()}"
            center_px = int(tracker.get_last_position()[0])
            center_py = int(tracker.get_last_position()[1])

            color = (0, 255, 0)
            if tracker.get_track_state() == 0:
                color = (0, 0, 255)
            cv2.putText(frame, tracker_name, (center_px, center_py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (center_px, center_py), 4, color, -1)


class Tracker:

    def __init__(self, track_id: int, max_age: int = None, track_current_info: dict = None):

        self.TRACK_ID = track_id
        track_current_info["tracker_id"] = self.TRACK_ID
        self.MAX_AGE = max_age
        self.state = 1  # 1: tracking, 0: waiting
        self.current_position = [
            track_current_info["px"], track_current_info["py"]]

        self.age = 0

        self.tracker_record = [
            track_current_info
        ]

    def get_track_id(self) -> int:
        return self.TRACK_ID

    def get_track_state(self) -> int:
        return self.state

    def set_position(self, track_current_info: dict = None) -> None:
        print(
            f"Tracker {self.TRACK_ID} is updated with new position: {track_current_info['px']}, {track_current_info['py']}")
        self.state = 1
        self.age = 0
        track_current_info["tracker_id"] = self.TRACK_ID
        self.tracker_record.append(track_current_info)
        self.current_position = [
            track_current_info["px"], track_current_info["py"]]

    def get_tracker_record(self) -> list:
        return self.tracker_record

    def calculate_distance(self, px, py) -> float:
        return ((px - self.current_position[0])**2 + (py - self.current_position[1])**2)**0.5

    def is_old(self) -> bool:
        return self.age > self.MAX_AGE

    def get_last_position(self):
        return self.current_position
