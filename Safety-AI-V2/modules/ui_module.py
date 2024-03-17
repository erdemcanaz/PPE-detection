import cv2
import copy, random
import numpy as np

class UIModule:
    def __init__(self):
        self.BACKGROUND_IMAGE = cv2.imread("images/src/backgrounds/ui_background.png")        
        self.PERSON_EMOJIS ={
            "green_forklift_hardhat": cv2.imread("images/src/icons/green_forklift_hardhat.png"),
            "green_forklift": cv2.imread("images/src/icons/green_forklift.png"),
            "green_hardhat_height": cv2.imread("images/src/icons/green_hardhat_height.png"), 
            "green_hardhat": cv2.imread("images/src/icons/green_hardhat.png"),
            "green_height": cv2.imread("images/src/icons/green_height.png"),
            "green": cv2.imread("images/src/icons/green.png"),
            "red_forklift_hardhat": cv2.imread("images/src/icons/red_forklift_hardhat.png"),
            "red_forklift": cv2.imread("images/src/icons/red_forklift.png"),
            "red_hardhat_height": cv2.imread("images/src/icons/red_hardhat_height.png"),
            "red_hardhat": cv2.imread("images/src/icons/red_hardhat.png"),
            "red_height": cv2.imread("images/src/icons/red_height.png"),
            "red": cv2.imread("images/src/icons/red.png"),
        }

        self.METER_TO_PIXEL_RATIO = 16.75 # 1 meter = 16.75 pixel
        self.ORIGIN_OFFSET_PX = (481, 905) # (x, y)
        self.UUID_COLORS = {} # {camera_uuid: (r, g, b)}

    
    def get_background_image(self):
        return copy.deepcopy(self.BACKGROUND_IMAGE)
    
    def get_person_emoji(self, is_violating:bool = False,  is_in_forklift:bool=False, is_wearing_hard_hat:bool = False, is_at_height:bool = False, scale_factor:float = 1):
        key_name = "red" if is_violating else "green"
        key_name = key_name + "_forklift" if is_in_forklift else key_name
        key_name = key_name + "_hardhat" if is_wearing_hard_hat else key_name
        key_name = key_name + "_height" if is_at_height else key_name

        emoji_frame = copy.deepcopy(self.PERSON_EMOJIS[key_name])      
        new_width = int(emoji_frame.shape[1] * scale_factor)
        new_height = int(emoji_frame.shape[0] * scale_factor)
        resized_emoji_frame = cv2.resize(emoji_frame, (new_width, new_height))

        return resized_emoji_frame
    
    def overlay_person_emoji(self, frame, emoji_frame, person_x_px, person_y_px):
            emoji_height, emoji_width = emoji_frame.shape[:2]

            # Calculate the top-left corner of where the emoji will be placed
            start_x = person_x_px - (emoji_width  // 2)
            start_y = person_y_px - (emoji_height // 2)
            end_x = start_x + emoji_width
            end_y = start_y + emoji_height

            # Check for bounds and adjust if necessary
            if start_x < 0: start_x = 0
            if start_y < 0: start_y = 0
            if end_x > frame.shape[1]: end_x = frame.shape[1]
            if end_y > frame.shape[0]: end_y = frame.shape[0]

            # Overlay the emoji on the new_frame image
            frame[start_y:end_y, start_x:end_x] = emoji_frame[0:(end_y - start_y), 0:(end_x - start_x)]


    def update_ui_frame(self, multiple_camera_evaluation_results: list[dict] =None, wait_time_ms:int = 0, window_scale_factor:float = 1, emoji_scale_factor = 1.5)-> bool:
        # Draw the UI frame
        new_frame = self.get_background_image()

        for evaluation_results in multiple_camera_evaluation_results:
            camera_uuid= evaluation_results["camera_uuid"]
            if camera_uuid not in self.UUID_COLORS:
                self.UUID_COLORS[camera_uuid] = (int(255 * random.uniform(0, 1)), int(255 * random.uniform(0, 1)), int(255 * random.uniform(0, 1)))
            uuid_color = self.UUID_COLORS[camera_uuid]

            number_of_forklifts = evaluation_results["number_of_forklifts"]
            number_of_persons = evaluation_results["number_of_persons"]
            person_evaluations = evaluation_results["person_evaluations"]

            for person_evaluation in person_evaluations:
                is_in_forklift = person_evaluation["is_in_forklift"]
                is_wearing_hard_hat = person_evaluation["is_wearing_hard_hat"]
                is_at_height = person_evaluation["is_at_height"]

                is_violating_restricted_area = person_evaluation["is_violating_restricted_area_rule"]
                is_violating_hard_hat_area = person_evaluation["is_violating_hard_hat_rule"]
                is_violating_height_rule = person_evaluation["is_violating_height_rule"]
                is_violating = is_violating_restricted_area or is_violating_hard_hat_area or is_violating_height_rule

                person_x = person_evaluation["world_coordinate"][0][0]
                person_y = person_evaluation["world_coordinate"][1][0]

                person_x_px = int(person_x * self.METER_TO_PIXEL_RATIO) + self.ORIGIN_OFFSET_PX[0]
                person_y_px = int(person_y * self.METER_TO_PIXEL_RATIO) + self.ORIGIN_OFFSET_PX[1]       

                emoji_scaler = 1.5*emoji_scale_factor if is_violating else emoji_scale_factor       
                emoji_frame = self.get_person_emoji(scale_factor= emoji_scaler, is_violating=is_violating, is_in_forklift=is_in_forklift, is_wearing_hard_hat=is_wearing_hard_hat, is_at_height=is_at_height)
                self.overlay_person_emoji(new_frame, emoji_frame, person_x_px, person_y_px)

        new_width = int(new_frame.shape[1] * window_scale_factor)
        new_height = int(new_frame.shape[0] * window_scale_factor)
        resized_new_frame = cv2.resize(new_frame, (new_width, new_height))
        cv2.imshow("ui_deneme", resized_new_frame)

        # Wait for a key press to close the window
        key = cv2.waitKey(wait_time_ms)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False
        return True
     

