import json
import cv2
import scripts.svg_editor as svg_editor

def generate_report_EN(folder_path = None, report_config = None):
    REGION_DATA = None
    with open(report_config["region_info_path"], 'r') as file:
        REGION_DATA = json.load(file)
    
    svg_creator_object = svg_editor.MultiSVGCreator()

    #cover page------------------------------------------------
    cover_page_svg_path = f"{folder_path}/svg_exports/page_1_cover_page.svg"
    svg_creator_object.create_new_drawing(cover_page_svg_path)

    COVER_PAGE_TEMPLATE_EN_PATH = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["COVER_PAGE_TEMPLATE_EN"][0]
    cv2_image = cv2.imread(COVER_PAGE_TEMPLATE_EN_PATH, cv2.IMREAD_UNCHANGED)
    
    svg_creator_object.embed_cv2_image_adjustable_resolution(
        filename = cover_page_svg_path, 
        insert= (0,0), size = svg_creator_object.get_size() , 
        cv2_image = cv2_image, 
        constant_proportions= True, 
        quality_factor= 2
    )


    #save all------------------------------------------------
    svg_creator_object.save_all()






