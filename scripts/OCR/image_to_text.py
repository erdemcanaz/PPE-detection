import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Update the path as per your installation

print(pytesseract.image_to_string(Image.open('secret_image.jpeg') , lang='tur'))

