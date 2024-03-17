import cv2
import copy

class UIModule:
    def __init__(self):
        self.UI_BACKGROUND_IMAGE_PATH = "images/src/ui_background.png"

        self.METER_TO_PIXEL_RATIO = 16.75 # 1 meter = 16.75 pixel
        self.ORIGIN_OFFSET_PX = (481, 905) # (x, y)
        self.BACKGROUND_IMAGE = cv2.imread(self.UI_BACKGROUND_IMAGE_PATH)

    def update_ui_frame(self, evaluation_results: dict =None, wait_time_ms:int = 0, scale_factor:float = 1)-> bool:
        # Draw the UI frame
        new_frame = copy.deepcopy(self.BACKGROUND_IMAGE)

        camera_uuid= evaluation_results["camera_uuid"]
        number_of_forklifts = evaluation_results["number_of_forklifts"]
        number_of_persons = evaluation_results["number_of_persons"]
        person_evaluations = evaluation_results["person_evaluations"]

        for person_evaluation in person_evaluations:
            is_in_forklift = person_evaluation["is_in_forklift"]
            is_weared_hard_hat = person_evaluation["is_wearing_hard_hat"]
            is_at_height = person_evaluation["is_at_height"]

            is_violating_restricted_area = person_evaluation["is_violating_restricted_area_rule"]
            is_violating_hard_hat_area = person_evaluation["is_violating_hard_hat_rule"]
            is_violating_height_rule = person_evaluation["is_violating_height_rule"]

            person_x = person_evaluation["world_coordinate"][0][0]
            person_y = person_evaluation["world_coordinate"][1][0]

            person_x_px = int(person_x * self.METER_TO_PIXEL_RATIO) + self.ORIGIN_OFFSET_PX[0]
            person_y_px = int(person_y * self.METER_TO_PIXEL_RATIO) + self.ORIGIN_OFFSET_PX[1]

            cv2.circle(new_frame, (person_x_px, person_y_px), 10, (0, 0, 255), -1)

        new_width = int(new_frame.shape[1] * scale_factor)
        new_height = int(new_frame.shape[0] * scale_factor)
        resized_new_frame = cv2.resize(new_frame, (new_width, new_height))
        cv2.imshow("ui_deneme", resized_new_frame)

        # Wait for a key press to close the window
        key = cv2.waitKey(wait_time_ms)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True
     

