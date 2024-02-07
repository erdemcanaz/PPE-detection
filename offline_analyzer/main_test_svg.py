from scripts.report_writer import ReportWriter
import datetime
# CONTENT FOR THE COVER PAGE ========================================

report_title = "Safety-AI Report"
sub_title = "This report was automatically created after analyzing the camera recordings of the 'BESAN factory - Koltuk Ambarı' area between 08:00-16:00 on 31.01.2024 with Safty-AI. The data in the report is intended to collect developmental insight in the field of occupational health and safety."
exported_by = "Erdem Canaz"
exported_by_work_title= "(Digital and ManEx GenNext under the supervision of Kadir Dolar)"
approved_by_and_presented_to = "The report was prepared with the knowledge of Anıl Kaya, Haydar Çamurdan and Kadir Dolar. Submitted for the information of SHE department and Akif Tufan Perçiner"

export_date = datetime.datetime.now().strftime("%d.%m.%Y")
# DRAW THE COVER PAGE ========================================
cover_page_writer = ReportWriter()

light_blue = cover_page_writer.get_color_by_name('light_blue')
dark_blue = cover_page_writer.get_color_by_name('dark_blue')
dark_blue_2 = cover_page_writer.get_color_by_name('dark_blue_2')
white = cover_page_writer.get_color_by_name('white')

cover_page_writer.add_text(
                              x=10, y=100, 
                              text=report_title, 
                              font_size='14', font_family='Oswald',
                              fill_color=dark_blue, stroke_color=dark_blue, stroke_width='0.75'
                          )

cover_page_writer.add_text_with_width_limit(
                              x=10, y=107.5, 
                              text = sub_title,
                              font_size='3', font_family='Oswald',
                              fill_color=light_blue, stroke_color=light_blue, stroke_width='0.0',
                              width_limit=100,
                              line_height=3.75
                            )

cover_page_writer.add_text_with_width_limit(
                              x=10, y=125, 
                              text = approved_by_and_presented_to,
                              font_size='3', font_family='Oswald',
                              fill_color=light_blue, stroke_color=light_blue, stroke_width='0.0',
                              width_limit=70,
                              line_height=3.25
                            )

cover_page_writer.add_text(
                              x=117.5, y=125, 
                              text=export_date, 
                              font_size='4.5', font_family='Century Gothic',
                              fill_color=dark_blue_2, stroke_color=dark_blue_2, stroke_width='0.0'
                          )

cover_page_writer.add_rotated_text(
                              x=205, y=85, rotate_angle=90,
                              text=exported_by, 
                              font_size='10', font_family='Oswald',
                              fill_color=white, stroke_color=white, stroke_width='0.0'
                            )

cover_page_writer.add_rotated_text(
                              x=210, y=85, rotate_angle=90,
                              text=exported_by_work_title, 
                              font_size='2.25', font_family='Oswald',
                              fill_color=light_blue, stroke_color=light_blue, stroke_width='0.0'
                            )

import cv2

# # Load the image as a cv2 frame
# image = cv2.imread('scripts/dummy_image.jpg')

# # Add the image to the cover page
# cover_page_writer.add_cv2_frame(image, x=0, y=100)

cover_page_writer.add_image_file('scripts/dummy_image.jpg', x=0, y=0)

cover_page_writer.export_modified_svg('secret_cover_page.svg')

