import datetime, json, uuid, pprint, io
import cv2
import scripts.svg_editor as svg_editor

import matplotlib.pyplot as plt
import numpy as np


def generate_report_EN(video_analyzer_object = None, folder_path = None, report_config = None, all_sorted_tracks:list[dict] = None, all_tracking_rows:list[dict] = None, all_sorted_hard_hat_rows:list[dict] = None, all_hard_hat_rows:list[dict] = None):
    REGION_DATA = None
    with open(report_config["region_info_path"], 'r') as file:
        REGION_DATA = json.load(file)
    
    svg_creator_object = svg_editor.MultiSVGCreator()
    page_no = 0
    #cover page------------------------------------------------
    cover_page_svg_path = f"{folder_path}/svg_exports/page_{page_no}_cover_page.svg"
    svg_creator_object.create_new_drawing(cover_page_svg_path, size=('1244px', '1756px'))

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
        svg_creator_object.create_new_drawing(regional_info_svg_path, size=('1244px', '1756px'))

        cv2_image = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        svg_creator_object.embed_cv2_image_adjustable_resolution(
            filename = regional_info_svg_path, 
            insert= (0,0), size = svg_creator_object.get_size() , 
            cv2_image = cv2_image, 
            constant_proportions= True, 
            quality_factor= 1
        )

        page_no += 1

    #restricted area violations-------------------------------
        
    #TODO: append a page related to how the restricted are violation is calculated
        
    #--------------
    if report_config["check_restricted_area_violation"]:        

        RESTRICTED_AREA_VIOLATION_TEMPLATE = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["RESTRICTED_AREA_TEMPLATE_EN"][0]
        cv2_image = cv2.imread(RESTRICTED_AREA_VIOLATION_TEMPLATE, cv2.IMREAD_UNCHANGED)

        for i in range(0, len(all_sorted_tracks), 3):
            #Render restricted area detection page
            restricted_area_page_n_svg_path = f"{folder_path}/svg_exports/page_{page_no}_restricted_area_violation_page.svg"
            svg_creator_object.create_new_drawing(restricted_area_page_n_svg_path, size=('1244px', '1756px'))

            svg_creator_object.embed_cv2_image_adjustable_resolution(
            filename = restricted_area_page_n_svg_path, 
            insert= (0,0), size = svg_creator_object.get_size() , 
            cv2_image = cv2_image, 
            constant_proportions= True, 
            quality_factor= 1
            )

            # This slice will get up to 3 elements, handling cases where there are fewer than 3 elements left
            current_tracks = all_sorted_tracks[i:i+3]
            for no, track_info in enumerate(current_tracks):
                if track_info is None:
                    continue
                first_frame_date = track_info["first_frame_date"] #a datetime.datetime object
                first_frame_index = track_info["first_frame_index"] 
                first_frame_time = track_info["first_frame_time"] #str video timestamp
                last_frame_index = track_info["last_frame_index"]
                last_frame_time = track_info["last_frame_time"]
                track_id = int(track_info["track_id"])
                track_violation_score = track_info["violation_score"]
        
                str_date = first_frame_date.strftime(f"%d.%m.%Y | %H:%M:%S ({first_frame_time})")
                x = []
                y = []

                for info_dict in all_tracking_rows:
                    if int(info_dict["tracker_id"]) == track_id:
                        x.append( float(info_dict["person_x"]) )
                        y.append( float(info_dict["person_y"]) )

                x_filtered = []
                y_filtered = []
                for j in range(len(x)):
                    if j+2 >= len(x):
                        break
                    
                    x_filtered.append((x[j] + x[j+1] + x[j+2])/3)
                    y_filtered.append((y[j] + y[j+1] + y[j+2])/3)
                # Load your background image
                bg_image = plt.imread(REGION_DATA["DEFAULT_TEMPLATE_PATHS"]['2D_MAP'][0])

                # Create the plot
                plt.figure(figsize=(8, 6))
                plt.imshow(bg_image, extent=[0,10 , 0, 10])  # Adjust extent as needed
                plt.scatter(x_filtered, y_filtered, color='black', s = 20)  # Plot data points on top of the background image
                plt.axis('on')  # You can turn this off with 'off' if you don't want the axis
                
                plt.plot(x, y, color='black', label='Connections')  # Connect the nodes

                # Highlight the initial and end nodes
                plt.scatter(x[0], y[0], color='green', s=60, edgecolor='black', label='Start')  # Initial node
                plt.scatter(x[-1], y[-1], color='red', s=60, edgecolor='black', label='End')  # End node
                plt.legend()  # Show legend to label start and end nodes

                #plt.show()

                # Save the plot to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Read the buffer with OpenCV
                img_buf_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                plot_as_cv2_image = cv2.imdecode(img_buf_arr, 1)
                
                plt.close()

                svg_creator_object.embed_cv2_image_adjustable_resolution(
                    filename = restricted_area_page_n_svg_path, 
                    insert= (100,215+471*no), size = ("400px", "400px") , 
                    cv2_image = plot_as_cv2_image, 
                    constant_proportions= True, 
                    quality_factor= 1
                )

                svg_creator_object.add_text(
                    filename = restricted_area_page_n_svg_path, 
                    text = str_date, 
                    insert= (125,235+471*no), 
                    fill_color = svg_creator_object.get_color('dark_blue'), 
                    stroke_color= svg_creator_object.get_color('dark_blue'), 
                    stroke_width=1,
                    font_size='24px'
                )

                svg_creator_object.add_text(
                    filename = restricted_area_page_n_svg_path, 
                    text = f"%{track_violation_score*100:.1f}", 
                    insert= (800,235+471*no), 
                    fill_color = svg_creator_object.get_color('dark_blue'), 
                    stroke_color= svg_creator_object.get_color('dark_blue'), 
                    stroke_width=1,
                    font_size='24px'
                )       

                svg_creator_object.add_text(
                    filename = restricted_area_page_n_svg_path, 
                    text = f"{i+1+no} / {len(all_sorted_tracks)}", 
                    insert= (1060,235+471*no), 
                    fill_color = svg_creator_object.get_color('dark_blue'), 
                    stroke_color= svg_creator_object.get_color('dark_blue'), 
                    stroke_width=1,
                    font_size='24px'
                )       

                frames_to_sample = [first_frame_index, (last_frame_index+first_frame_index)//2, last_frame_index ]
                for counter, frame_index in enumerate(frames_to_sample):
                    video_analyzer_object.set_current_frame_index(frame_index)
                    frame = video_analyzer_object.get_current_frame()

                    kernel_size = report_config["report_blur_kernel_size"]
                    blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                    svg_creator_object.embed_cv2_image_adjustable_resolution(
                        filename = restricted_area_page_n_svg_path, 
                        insert= (475+275*counter,300+471*no), size = ("250px", "250px") , 
                        cv2_image = blurred_frame, 
                        constant_proportions= True, 
                        quality_factor= 1
                    )                                    

            page_no += 1


    #Hard-hat violations-------------------------------------
    if report_config["check_hard_hat_violation"]: 

        HARD_HAT_VIOLATION_TEMPLATE = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["HARD_HAT_TEMPLATE_EN"][0]
        cv2_image = cv2.imread(HARD_HAT_VIOLATION_TEMPLATE, cv2.IMREAD_UNCHANGED)
        
        number_of_images_per_page = 24
        for i in range(0, len(all_sorted_hard_hat_rows), number_of_images_per_page):

            hard_hat_violation_page_n_svg_path = f"{folder_path}/svg_exports/page_{page_no}_hard_hat_violation_page.svg"
            svg_creator_object.create_new_drawing(hard_hat_violation_page_n_svg_path, size=('1244px', '1756px'))

            svg_creator_object.embed_cv2_image_adjustable_resolution(
                filename = hard_hat_violation_page_n_svg_path, 
                insert= (0,0), size = svg_creator_object.get_size() , 
                cv2_image = cv2_image, 
                constant_proportions= True, 
                quality_factor= 1
            )
             
            hard_hat_violation_batch = all_sorted_hard_hat_rows[i:i+number_of_images_per_page]
            for counter, hard_hat_violation_info in enumerate(hard_hat_violation_batch):
                row_no = counter // 4
                column_no = counter % 4

                bbox_confidence = hard_hat_violation_info["bbox_confidence"]
                bbox_coordinates = hard_hat_violation_info["bbox_coordinates"]
                current_second =hard_hat_violation_info["current_second"]
                hard_hat_detection_date = hard_hat_violation_info["date"]
                frame_index = hard_hat_violation_info["frame_index"]
                is_safety_equipment_present = hard_hat_violation_info["is_safety_equipment_present"]
                safety_equipment_bbox_center = hard_hat_violation_info["safety_equipment_bbox_center"]
                safety_equipment_class = hard_hat_violation_info["safety_equipment_class"]
                safety_equipment_confidence = hard_hat_violation_info["safety_equipment_confidence"]
                video_time = hard_hat_violation_info["video_time"]
                violation_score = hard_hat_violation_info["violation_score"]

                video_analyzer_object.set_current_frame_index(frame_index)
                frame = video_analyzer_object.get_current_frame()
                frame_section = frame[bbox_coordinates[1]:bbox_coordinates[3], bbox_coordinates[0]:bbox_coordinates[2]]
                
                str_date = hard_hat_detection_date.strftime(f"%d.%m.%Y | %H:%M:%S ({video_time})")

                svg_creator_object.add_text(
                    filename = hard_hat_violation_page_n_svg_path, 
                    text = f"%{violation_score*100:.1f} ({i}/{len(all_sorted_hard_hat_rows)})", 
                    insert=(50 +250*column_no, (200 + 250 * row_no)-2),
                    fill_color = svg_creator_object.get_color('dark_blue'), 
                    stroke_color= svg_creator_object.get_color('dark_blue'), 
                    stroke_width=1,
                    font_size='10px'
                )

                svg_creator_object.add_text(
                    filename = hard_hat_violation_page_n_svg_path, 
                    text = f"{str_date}", 
                    insert=(50 +250*column_no, (200 + 250 * row_no) -15),
                    fill_color = svg_creator_object.get_color('dark_blue'), 
                    stroke_color= svg_creator_object.get_color('dark_blue'), 
                    stroke_width=1,
                    font_size='10px'
                )

                kernel_size = report_config["report_blur_kernel_size"]
                blurred_frame = cv2.GaussianBlur(frame_section, (kernel_size, kernel_size), 0)                
                svg_creator_object.embed_cv2_image_adjustable_resolution(
                    filename=hard_hat_violation_page_n_svg_path,
                    insert=(50 +250*column_no, 200 + 250 * row_no),
                    size=("200px", "200px"),
                    cv2_image=blurred_frame,
                    constant_proportions=True,
                    quality_factor=1
                )

            page_no += 1

    #table of contents----------------------------------------
    page_no = 1
    
    content_page_1_svg_path = f"{folder_path}/svg_exports/page_{page_no}_table_of_contents.svg"
    svg_creator_object.create_new_drawing(content_page_1_svg_path, size=('1244px', '1756px'))

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
    svg_creator_object.create_new_drawing(content_page_2_svg_path, size=('1244px', '1756px'))

    CONTENT_PAGE_2_TEMPLATE_EN_PATH = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["CONTENT_PAGE_TEMPLATE_EN"][0] = REGION_DATA["DEFAULT_TEMPLATE_PATHS"]["CONTENT_PAGE_TEMPLATE_EN"][1]
    cv2_image = cv2.imread(CONTENT_PAGE_2_TEMPLATE_EN_PATH, cv2.IMREAD_UNCHANGED)
    svg_creator_object.embed_cv2_image_adjustable_resolution(
        filename = content_page_2_svg_path, 
        insert= (0,0), size = svg_creator_object.get_size() , 
        cv2_image = cv2_image, 
        constant_proportions= True, 
        quality_factor= 2
    )    

    #save all svg files------------------------------------------------
    svg_creator_object.save_all()
    

    #Convert to PDF----------------------------------------------------

    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    import os


    svg_folder = f"{folder_path}/svg_exports/"  # Adjust this to the path where your SVG files are located
    output_pdf_file = f"{folder_path}/{report_config['REPORT_NAME']}"  # The name of the combined PDF file you want to create

    # List all SVG files following the naming convention "page_i_......."
    svg_files = sorted([f for f in os.listdir(svg_folder) if f.startswith('page_') and f.endswith('.svg')],
                    key=lambda x: int(x.split('_')[1]))
                    
    page_width, page_height = A4  # A4 size is 595 x 842 points by default
    
    c = canvas.Canvas(output_pdf_file, pagesize=A4)
    
    for svg_file in svg_files:
        drawing = svg2rlg(os.path.join(svg_folder, svg_file))
        
        # Original SVG dimensions
        original_width_px, original_height_px = 1244,1756  # Known SVG dimensions in pixels
        # Convert pixels to points for ReportLab (1 point = 1.33333 pixels approximately)
        original_width = original_width_px * (72 / 96)
        original_height = original_height_px * (72 / 96)
        
        # Calculate scale to fit the SVG in the A4 page size
        scale_width = page_width / original_width
        scale_height = page_height / original_height
        scale = min(scale_width, scale_height)
        scale = scale * 0.75
        
        # Apply scaling
        drawing.width, drawing.height = drawing.width * scale, drawing.height * scale
        drawing.scale(scale, scale)

        # Center the drawing (optional)
        x_position = (page_width - drawing.width) / 2
        y_position = (page_height - drawing.height) / 2
        
        renderPDF.draw(drawing, c, x_position, y_position)
        
        # Create a new page for the next SVG
        c.showPage()
    
    c.save()
    print(f"Combined PDF saved as: {output_pdf_file}")







