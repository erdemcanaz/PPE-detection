from scripts.report_writer import ReportWriter


report_writer_object = ReportWriter()

light_blue = report_writer_object.get_color_by_name('light_blue')
report_writer_object.add_text(
                              x=10, y=97.5, 
                              text="Safety AI Report", 
                              font_size='14', font_family='Arial',
                              fill_color=light_blue, stroke_color=light_blue, stroke_width='1'
                            )
report_writer_object.export_modified_svg('cover_page.svg')