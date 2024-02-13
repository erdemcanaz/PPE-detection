import base64
import svgwrite
import cv2
from io import BytesIO
from PIL import Image

class MultiSVGCreator:
    def __init__(self):
        self.drawings = {}

    COLOR_PALETTE = {
        "light_blue": "rgb(125, 206, 237)",
        "dark_blue": "rgb(0, 96, 169)",
        "dark_blue_2": "rgb(0, 51, 204)",
        "black": "rgb(0, 0, 0)",
        "white": "rgb(255, 255, 255)"
    }

    #functionalities
    def create_new_drawing(self, filename:str=None, size=('1244px', '1756px')):
        """        
        Create or modify an SVG drawing. Resizes the existing SVG if it exists, or creates a new one.

        :param filename: The name of the file to save the SVG as.
        :param size: The size of the SVG canvas.
        #300DPI : size=('2480px', '3508px')
        #150DPI : size=('1244px', '1756px')
        """
        if filename in self.drawings:
            raise Exception(f"There is already an svg file named as {filename}")        

        self.SIZE = size
        self.drawings[filename] = svgwrite.Drawing(filename=filename, size=size)     

    def save_all(self):
        """
        Save all SVG drawings to their respective files.
        """
        for drawing in self.drawings.values():
            drawing.save()

    #getters and setters
    def get_size(self):
        return self.SIZE
    
    def get_color(self, color_name):
        return MultiSVGCreator.COLOR_PALETTE[color_name]
    
    #geometry
    def add_rectangle(self, filename, insert, size, fill='none', stroke='black'):
        """
        Add a rectangle to an SVG drawing.
        
        :param filename: The filename of the SVG to add the rectangle to.
        :param insert: A tuple (x, y) for the top-left corner of the rectangle.
        :param size: A tuple (width, height) for the size of the rectangle.
        :param fill: The fill color of the rectangle.
        :param stroke: The stroke color of the rectangle.
        """
        if filename in self.drawings:
            self.drawings[filename].add(self.drawings[filename].rect(insert=insert, size=size, fill=fill, stroke=stroke))

    def add_circle(self, filename, center, radius, fill='none', stroke='black'):
        """
        Add a circle to an SVG drawing.
        
        :param filename: The filename of the SVG to add the circle to.
        :param center: A tuple (cx, cy) for the center of the circle.
        :param radius: The radius of the circle.
        :param fill: The fill color of the circle.
        :param stroke: The stroke color of the circle.
        """
        if filename in self.drawings:
            self.drawings[filename].add(self.drawings[filename].circle(center=center, r=radius, fill=fill, stroke=stroke))

    #image
    def link_image(self, filename, insert, size, href):
        """
        Link an image to an SVG drawing.
        
        :param filename: The filename of the SVG to add the image to.
        :param insert: A tuple (x, y) for the top-left corner of the image.
        :param size: A tuple (width, height) for the size of the image.
        :param href: The path or URL to the image file.
        """
        if filename in self.drawings:
            self.drawings[filename].add(self.drawings[filename].image(href=href, insert=insert, size=size))

    def embed_image(self, filename, insert, size, href):
        """
        Embed an image into an SVG drawing.

        :param filename: The filename of the SVG to add the image to.
        :param insert: A tuple (x, y) for the top-left corner of the image.
        :param size: A tuple (width, height) for the size of the image.
        :param href: The path to the image file.
        """
        if filename in self.drawings:
            # Read the image file and encode it as base64
            with open(href, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine the image's MIME type based on its extension
            # You may need to add other image formats as necessary
            mime_type = "image/jpeg"  # Default to JPEG
            if href.lower().endswith('.png'):
                mime_type = "image/png"
            elif href.lower().endswith('.gif'):
                mime_type = "image/gif"
            elif href.lower().endswith('.svg'):
                mime_type = "image/svg+xml"
            
            # Construct the data URI
            data_uri = f"data:{mime_type};base64,{image_data}"
            
            # Embed the image data in the SVG
            self.drawings[filename].add(self.drawings[filename].image(href=data_uri, insert=insert, size=size))
    
    def embed_image_low_resolution(self, filename, insert, size, href):
        """
        Embed a lower resolution image into an SVG drawing, using updated Pillow resampling method.

        :param filename: The filename of the SVG to add the image to.
        :param insert: A tuple (x, y) for the top-left corner of the image.
        :param size: A tuple (width, height) in pixels for the target size of the image.
        :param href: The path to the image file.
        """
        if filename in self.drawings:
            # Open and resize the image to the desired dimensions
            with Image.open(href) as img:
                # Convert size from SVG dimensions (which may include units like 'px') to integers
                target_size = (int(size[0].replace('px', '')), int(size[1].replace('px', '')))
                
                # Use Image.Resampling.LANCZOS for high-quality downsampling
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert the resized image to bytes
                img_byte_arr = BytesIO()
                img_format = img.format if img.format else 'JPEG'  # Default format if not detected
                resized_img.save(img_byte_arr, format=img_format)
                
                # Encode the image file as base64
                image_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Determine the image's MIME type
            mime_type = f"image/{img_format.lower()}"
            
            # Construct the data URI
            data_uri = f"data:{mime_type};base64,{image_data}"
            
            # Embed the resized and encoded image data in the SVG
            self.drawings[filename].add(self.drawings[filename].image(href=data_uri, insert=insert, size=size))
            
    def embed_cv2_image(self, filename, insert, size, cv2_image):
        """
        Embed an OpenCV image into an SVG drawing.

        :param filename: The filename of the SVG to add the image to.
        :param insert: A tuple (x, y) for the top-left corner of the image.
        :param size: A tuple (width, height) for the size of the image.
        :param cv2_image: The OpenCV image (NumPy array).
        """
        if filename in self.drawings:
            # Convert the OpenCV image to PNG format in memory
            _, buffer = cv2.imencode('.png', cv2_image)
            
            # Encode the PNG image data as base64
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Construct the data URI
            data_uri = f"data:image/png;base64,{image_data}"
            
            # Embed the image data in the SVG
            self.drawings[filename].add(self.drawings[filename].image(href=data_uri, insert=insert, size=size))

    def embed_cv2_image_adjustable_resolution(self, filename, insert, size, cv2_image, constant_proportions = False, quality_factor = 1.0):
        """
        Embed a lower resolution OpenCV image into an SVG drawing.

        :param filename: The filename of the SVG to add the image to.
        :param insert: A tuple (x, y) for the top-left corner of the image.
        :param size: A tuple (width, height) in pixels for the target size of the image.
        :param cv2_image: The OpenCV image (NumPy array).
        :constant_proportions: Whether the image proportions are preserved or not while resizing
        :quality_factor: by increasing this value, the quality of the embeded image is increased but the size of the document is increased.
        """

        if filename in self.drawings:
            # Convert size from SVG dimensions (which may include units like 'px') to integers
            target_width = int(size[0].replace('px', ''))
            target_height = int(size[1].replace('px', ''))
        else:
            raise Exception(f"Drawing with filename {filename} does not exist.")
        
        resized_image = None
        height, width = cv2_image.shape[:2]
        
        if constant_proportions:
            # Calculate the scale factor to maintain aspect ratio
            height, width = cv2_image.shape[:2]
            scale_w = target_width / width
            scale_h = target_height / height
            scale = min(scale_w, scale_h)

            # Calculate new dimensions
            new_width = int(quality_factor * width * scale)
            new_height = int(quality_factor * height * scale)

            # Resize the image
            resized_image = cv2.resize(cv2_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Resize without preserving aspect ratio
            target_width = int(quality_factor * target_width)
            target_height = int(quality_factor * target_height)
            resized_image = cv2.resize(cv2_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)        
            
        # Convert the resized OpenCV image to PNG format in memory
        _, buffer = cv2.imencode('.png', resized_image)
        
        # Encode the PNG image data as base64
        image_data = base64.b64encode(buffer).decode('utf-8')
        
        # Construct the data URI
        data_uri = f"data:image/png;base64,{image_data}"
        
        # Embed the image data in the SVG
        self.drawings[filename].add(self.drawings[filename].image(href=data_uri, insert=insert, size=size))

    #text
    def add_text(self, filename, insert, text, font_size='20px', font_family='Arial', fill_color='rgb(0, 0, 0)', stroke_color='rgb(0, 0, 0)', stroke_width=1.5, rotation_angle=0):
        """
        Adds text to an SVG drawing using the svgwrite library, with an option to rotate the text.

        :param filename: The filename of the SVG to add the text to.
        :param insert: Tuple of the bottom-left x and y-coordinate for the text's position as a tuple (x,y).
        :param text: The text content to add.
        :param font_size: The font size of the text.
        :param font_family: The font family of the text.
        :param fill_color: The fill color of the text.
        :param stroke_color: The stroke color of the text.
        :param stroke_width: The stroke width of the text.
        :param rotation_angle: The rotation angle of the text in degrees.(CW positive, CCW negative)
        """
        if filename not in self.drawings:
            print(f"Drawing with filename {filename} does not exist.")
            return

        # Construct the style string with fill color, stroke color, and stroke width
        style = f'font-size: {font_size}; font-family: {font_family}; fill: {fill_color}; stroke: {stroke_color}; stroke-width: {stroke_width}px;'

        # If a rotation is specified, add a rotation transform
        transform = f'rotate({-rotation_angle}, {insert[0]}, {insert[1]})'

        # Create the text element with the specified attributes, style, and transformation
        text_element = svgwrite.text.Text(text, insert=insert, style=style, transform=transform)

        # Add the text element to the drawing
        self.drawings[filename].add(text_element)

    def add_text_with_width_limit(self, filename, insert, text, font_size='20px', font_family='Arial', fill_color='rgb(0, 0, 0)', stroke_color='rgb(0, 0, 0)', stroke_width=1.5, width_limit=100, line_height=20):
        """
        Adds text to an SVG drawing with a width limit, breaking the text into multiple lines if necessary.

        :param filename: The filename of the SVG to add the text to.
        :param insert: tuple of the bottom-left x and y -coordinate for the first line's position as a tuple (x,y)
        :param text: The text content to add.
        :param font_size: The font size of the text.
        :param font_family: The font family of the text.
        :param fill_color: The fill color of the text.
        :param stroke_color: The stroke color of the text.
        :param stroke_width: The stroke width of the text.
        :param width_limit: The maximum number of characters in a line before wrapping.
        :param line_height: The height of each line of text.
        """
        if filename not in self.drawings:
            print(f"Drawing with filename {filename} does not exist.")
            return

        # Style string
        style = f'font-size: {font_size}; font-family: {font_family}; fill: {fill_color}; stroke: {stroke_color}; stroke-width: {stroke_width}px;'

        # Initialize variables to hold lines of text
        lines = []
        line = ""
        counter = 0

        # Break text into lines based on width_limit
        for char in text:
            line += char
            counter += 1
            if counter >= width_limit and char == " ":
                lines.append(line.strip())
                line = ""
                counter = 0
        if line:
            lines.append(line.strip())  # Add any remaining text as a new line

        # Add lines of text to the SVG
        text_element = svgwrite.text.Text("", insert=insert, style=style)
        dy = 0
        for line_text in lines:
            tspan = svgwrite.text.TSpan(line_text, x=[insert[0]], dy=[dy])
            text_element.add(tspan)
            dy = line_height  # Increment y position for each new line

        self.drawings[filename].add(text_element)


# Usage example            
if __name__ == '__main__':
    svg_creator = MultiSVGCreator()
    svg_creator.create_new_drawing(filename= 'rendered_svg_files/page_1.svg')
    #svg_creator.add_circle('example1.svg', (100, 100), 80, fill='green', stroke='orange')

    cover_page_template_href = "C:\\Users\\Levovo20x\\Documents\\GitHub\\SVG-simple\\templates\\cover_page_template_300DPI.png"
    cv2_image = cv2.imread(cover_page_template_href, cv2.IMREAD_UNCHANGED)
    svg_creator.embed_cv2_image_adjustable_resolution(filename = 'rendered_svg_files/page_1.svg', insert= (0,0), size = svg_creator.get_size() , cv2_image = cv2_image, constant_proportions= True, quality_factor= 2)
    #svg_creator.embed_image(filename = 'rendered_svg_files/page_1.svg', insert=(0,0), size=svg_creator.get_size(), href = cover_page_template_href)

    href= input("Enter the path of the image: ")
    svg_creator.embed_image(filename='rendered_svg_files/page_1.svg', insert=(0,0),size=('100px', '100px'), href=href)
    svg_creator.embed_image_low_resolution(filename='rendered_svg_files/page_1.svg', insert=(0,100),size=('100px', '100px'), href=href)

    cv2_image = cv2.imread("nature.png", cv2.IMREAD_COLOR)
    svg_creator.embed_cv2_image(filename = 'rendered_svg_files/page_1.svg', insert= (0,200), size =('100px', '100px'), cv2_image = cv2_image)
    svg_creator.embed_cv2_image_adjustable_resolution(filename = 'rendered_svg_files/page_1.svg', insert= (0,300), size =('100px', '100px'), cv2_image = cv2_image, constant_proportions= True, quality_factor= 3)


    svg_creator.add_text(filename = 'rendered_svg_files/page_1.svg', text = "deneme", insert= (0,80), fill_color = svg_creator.get_color('dark_blue'), stroke_color= svg_creator.get_color('dark_blue'), stroke_width=1)
    long_text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Nunc scelerisque viverra mauris in. Volutpat commodo sed egestas egestas fringilla phasellus. Aliquam sem fringilla ut morbi tincidunt augue interdum. Ac placerat vestibulum lectus mauris ultrices eros in cursus turpis. Luctus accumsan tortor posuere ac ut consequat semper viverra nam. Convallis posuere morbi leo urna. Mattis molestie a iaculis at erat pellentesque. Sapien et ligula ullamcorper malesuada proin libero nunc. Ac tortor dignissim convallis aenean et tortor. Id aliquet lectus proin nibh nisl. Tempor id eu nisl nunc mi. Ipsum suspendisse ultrices gravida dictum fusce ut placerat orci nulla. Id consectetur purus ut faucibus pulvinar elementum integer. Adipiscing vitae proin sagittis nisl rhoncus mattis.
    """
    svg_creator.add_text_with_width_limit(filename =  'rendered_svg_files/page_1.svg', insert =(300,500), text = long_text, font_size='20px', font_family='Arial', fill_color='rgb(0, 0, 0)', stroke_color='rgb(0, 0, 0)', stroke_width=1.5, width_limit=100, line_height=20)
    svg_creator.save_all()
