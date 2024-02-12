import datetime, json, uuid
import cv2
import scripts.svg_editor as svg_editor

def generate_report_EN(folder_path = None, report_config = None):
    REGION_DATA = None
    with open(report_config["region_info_path"], 'r') as file:
        REGION_DATA = json.load(file)
    
    svg_creator_object = svg_editor.MultiSVGCreator()
    page_no = 0
    #cover page------------------------------------------------
    cover_page_svg_path = f"{folder_path}/svg_exports/page_{page_no}_cover_page.svg"
    svg_creator_object.create_new_drawing(cover_page_svg_path, size=('1244', '1756px'))

    COVER_PAGE_TEMPLATE_EN_PATH = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["COVER_PAGE_TEMPLATE_EN"][0]
    cv2_image = cv2.imread(COVER_PAGE_TEMPLATE_EN_PATH, cv2.IMREAD_UNCHANGED)
    
    svg_creator_object.embed_cv2_image_adjustable_resolution(
        filename = cover_page_svg_path, 
        insert= (0,0), size = svg_creator_object.get_size() , 
        cv2_image = cv2_image, 
        constant_proportions= True, 
        quality_factor= 2
    )

    svg_creator_object.add_text(
        filename = cover_page_svg_path, 
        insert = (80, 575), 
        text = report_config["TEXTUAL_INFO"]["COVER_PAGE"]["REPORT_TITLE"], 
        font_size='100px', 
        font_family='Times New Roman', 
        fill_color=svg_creator_object.get_color("dark_blue"), 
        stroke_color=svg_creator_object.get_color("dark_blue"), 
        stroke_width=2.25,
        rotation_angle = 0
    )

    svg_creator_object.add_text_with_width_limit(
        filename = cover_page_svg_path, 
        insert=(80,620), 
        text=report_config["TEXTUAL_INFO"]["COVER_PAGE"]["REPORT_PURPOSE"], 
        font_size='20px', 
        font_family='Times New Roman',  
        fill_color=svg_creator_object.get_color("light_blue"), 
        stroke_color=svg_creator_object.get_color("dark_blue"), 
        stroke_width=0, 
        width_limit=85, 
        line_height=20
    )

    svg_creator_object.add_text_with_width_limit(
        filename = cover_page_svg_path, 
        insert=(80,725), 
        text=report_config["TEXTUAL_INFO"]["COVER_PAGE"]["RELATED_PEOPLE"], 
        font_size='20px', 
        font_family='Times New Roman',  
        fill_color=svg_creator_object.get_color("light_blue"), 
        stroke_color=svg_creator_object.get_color("light_blue"), 
        stroke_width=0, 
        width_limit=60, 
        line_height=20
    )

    report_generated_at = datetime.datetime.now().strftime("%d.%m.%Y")
    svg_creator_object.add_text(
        filename = cover_page_svg_path, 
        insert = (640, 750), 
        text = report_generated_at, 
        font_size='40', 
        font_family='Times New Roman', 
        fill_color=svg_creator_object.get_color("dark_blue"), 
        stroke_color=svg_creator_object.get_color("dark_blue"), 
        stroke_width=0.5,
        rotation_angle = 0
    )

    svg_creator_object.add_text(
        filename = cover_page_svg_path, 
        insert = (1200, 510), 
        text = report_config["TEXTUAL_INFO"]["COVER_PAGE"]["EXPORTED_BY"],
        font_size='60', 
        font_family='Times New Roman', 
        fill_color=svg_creator_object.get_color("white"), 
        stroke_color=svg_creator_object.get_color("white"), 
        stroke_width=1,
        rotation_angle = 90
    )

    uuid_4 = uuid.uuid4()
    upper_case_uuid4 = str(uuid_4).upper()
    svg_creator_object.add_text(
        filename = cover_page_svg_path, 
        insert = (975, 20), 
        text = upper_case_uuid4,
        font_size='12', 
        font_family='Times New Roman', 
        fill_color=svg_creator_object.get_color("dark_blue"), 
        stroke_color=svg_creator_object.get_color("dark_blue"), 
        stroke_width=0.1,
        rotation_angle = 0
    )

    page_no += 1

    #2 pages are reserved for the table of contents
    page_no += 1
    page_no += 1

    #regional_info-------------------------------------------
    region_specific_info_templates = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["REGIONAL_INFO_TEMPLATE_EN"]
    for template_path in region_specific_info_templates:
        regional_info_svg_path = f"{folder_path}/svg_exports/page_{page_no}_regional_info.svg"
        svg_creator_object.create_new_drawing(regional_info_svg_path, size=('1244', '1756px'))

        cv2_image = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        svg_creator_object.embed_cv2_image_adjustable_resolution(
            filename = regional_info_svg_path, 
            insert= (0,0), size = svg_creator_object.get_size() , 
            cv2_image = cv2_image, 
            constant_proportions= True, 
            quality_factor= 2
        )

        page_no += 1

    #table of contents------------------------------------------------
    page_no = 1

    content_page_1_svg_path = f"{folder_path}/svg_exports/page_{page_no}_table_of_contents.svg"
    svg_creator_object.create_new_drawing(content_page_1_svg_path, size=('1244', '1756px'))

    CONTENT_PAGE_1_TEMPLATE_EN_PATH = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["CONTENT_PAGE_TEMPLATE_EN"][0]
    cv2_image = cv2.imread(CONTENT_PAGE_1_TEMPLATE_EN_PATH, cv2.IMREAD_UNCHANGED)
    svg_creator_object.embed_cv2_image_adjustable_resolution(
        filename = content_page_1_svg_path, 
        insert= (0,0), size = svg_creator_object.get_size() , 
        cv2_image = cv2_image, 
        constant_proportions= True, 
        quality_factor= 2
    )

    page_no += 1
    #----------
    content_page_2_svg_path = f"{folder_path}/svg_exports/page_{page_no}_table_of_contents.svg"
    svg_creator_object.create_new_drawing(content_page_2_svg_path, size=('1244', '1756px'))

    CONTENT_PAGE_2_TEMPLATE_EN_PATH = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["CONTENT_PAGE_TEMPLATE_EN"][0] = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["CONTENT_PAGE_TEMPLATE_EN"][1]
    cv2_image = cv2.imread(CONTENT_PAGE_2_TEMPLATE_EN_PATH, cv2.IMREAD_UNCHANGED)
    svg_creator_object.embed_cv2_image_adjustable_resolution(
        filename = content_page_2_svg_path, 
        insert= (0,0), size = svg_creator_object.get_size() , 
        cv2_image = cv2_image, 
        constant_proportions= True, 
        quality_factor= 2
    )    

    #save all------------------------------------------------
    svg_creator_object.save_all()






