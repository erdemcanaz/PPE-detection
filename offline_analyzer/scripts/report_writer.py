import xml.etree.ElementTree as ET

class ReportWriter:
    TEMPLATE_PATHS = {
            'cover_page_template': 'scripts/svg_templates/cover_page_template.svg',
    }
    COLOR_PALETTE = {
        "light_blue": "rgb(125, 206, 237)",
        "black": "rgb(0, 0, 0)",
        "white": "rgb(255, 255, 255)"
    }

    def __init__(self):
        self.tree = ET.parse(ReportWriter.TEMPLATE_PATHS['cover_page_template'])
        self.root = (self.tree).getroot()

    def add_text(self, x:int, y:int, text:str, font_size:str='20px', font_family:str='Arial', fill_color:str='rgb(0, 0, 0)', stroke_color:str='rgb(0, 0, 0)', stroke_width:float=1.5):
        # Construct the style string with fill color, stroke color, and stroke width
        style = f'font-size: {font_size}; font-family: {font_family}; fill: {fill_color}; stroke: {stroke_color}; stroke-width: {stroke_width};'
        
        # Create the text element with the constructed style
        text_element = ET.Element('{http://www.w3.org/2000/svg}text', x=str(x), y=str(y))
        text_element.text = text
        text_element.set('style', style)
        self.root.append(text_element)

    def export_modified_svg(self, output_file_name):
        (self.tree).write(output_file_name) # output_file_name = 'cover_page.svg'

    def get_color_by_name(self,color_name:str):
        return ReportWriter.COLOR_PALETTE[color_name]



    


