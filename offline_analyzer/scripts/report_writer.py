import cv2
import base64
import xml.etree.ElementTree as ET
from io import BytesIO

class ReportWriter:
    TEMPLATE_PATHS = {
            'cover_page_template': 'scripts/secret_svg_templates/cover_page_template.svg',
    }
    COLOR_PALETTE = {
        "light_blue": "rgb(125, 206, 237)",
        "dark_blue": "rgb(0, 96, 169)",
        "dark_blue_2": "rgb(0, 51, 204)",
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

    def add_text_with_width_limit(self, x:int, y:int, text:str, font_size:str='20px', font_family:str='Arial', fill_color:str='rgb(0, 0, 0)', stroke_color:str='rgb(0, 0, 0)', stroke_width:float=1.5, width_limit:int=100, line_height:int=20):
        # Construct the style string with fill color, stroke color, and stroke width
        style = f'font-size: {font_size}; font-family: {font_family}; fill: {fill_color}; stroke: {stroke_color}; stroke-width: {stroke_width};'
        
        lines = []
        line = ""
        counter = 0

        for char in text:
            line += char
            counter += 1
            if counter >= width_limit and char == " ":
                lines.append(line)
                line = ""
                counter = 0
        # Add any remaining text as a new line
        if line:
            lines.append(line)

        # Create text element and append tspans for each line
        text_element = ET.Element('{http://www.w3.org/2000/svg}text', x=str(x), y=str(y))
        text_element.set('style', style)
        dy = 0  # Initial offset for the first line
        for line in lines:
            tspan = ET.SubElement(text_element, '{http://www.w3.org/2000/svg}tspan', x=str(x), dy=str(dy))
            tspan.text = line
            dy = line_height  # Subsequent lines offset by line_height
        self.root.append(text_element)

    def add_rotated_text(self, x:int, y:int, rotate_angle:float, text:str, font_size:str='20px', font_family:str='Arial', fill_color:str='rgb(0, 0, 0)', stroke_color:str='rgb(0, 0, 0)', stroke_width:float=1.5):
        # Construct the style string with fill color, stroke color, and stroke width
        style = f'font-size: {font_size}; font-family: {font_family}; fill: {fill_color}; stroke: {stroke_color}; stroke-width: {stroke_width};'
        
        # Create the text element with the constructed style
        text_element = ET.Element('{http://www.w3.org/2000/svg}text', x=str(x), y=str(y))
        text_element.text = text
        text_element.set('style', style)
        
        # Set transform attribute for rotation to make the text vertical
        text_element.set('transform', f'rotate({-rotate_angle}, {x}, {y})')
        
        self.root.append(text_element)

    def add_cv2_frame(self, frame, x, y):        
        # Convert the frame to JPEG format in memory
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            print("Could not convert frame to JPEG format")
            return

        # Encode the JPEG image data to Base64
        encoded_image = base64.b64encode(buffer).decode("utf-8")
        
        # Ensure the namespace for href is correctly used
        nsmap = {'xlink': 'http://www.w3.org/1999/xlink'}
        ET.register_namespace('xlink', nsmap['xlink'])

        # Correctly set the href with namespace for the image element
        image_attributes = {
        'href': f"data:image/jpeg;base64,{encoded_image}",
        'x': str(x),
        'y': str(y),
        'width': str(frame.shape[1]),
        'height': str(frame.shape[0])
        }
        image_element = ET.SubElement(self.root, 'image', image_attributes)
    
    def add_image_file(self, image_path, x, y, width=None, height=None):
        # Read the image file in binary mode
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Encode the image data to Base64
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Determine the image MIME type based on the file extension
        # This is a simplistic approach; you might want to use a more robust method
        # in production code.
        mime_type = 'image/jpeg'  # Default to JPEG
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.svg'):
            mime_type = 'image/svg+xml'

        # Create a new 'image' element for the SVG with the Base64-encoded data
        image_attributes = {
            'href': f"data:{mime_type};base64,{encoded_image}",
            'x': str(x),
            'y': str(y)
        }

        # If width and height are provided, set them
        if width and height:
            image_attributes['width'] = str(width)
            image_attributes['height'] = str(height)

        ET.SubElement(self.root, 'image', image_attributes)

    def export_modified_svg(self, output_file_name):
        (self.tree).write(output_file_name) # output_file_name = 'cover_page.svg'

    def get_color_by_name(self,color_name:str):
        return ReportWriter.COLOR_PALETTE[color_name]



    


