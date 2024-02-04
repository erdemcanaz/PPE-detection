import os
import img2pdf
from PIL import Image

def convert_images_to_pdf(folder_path, output_pdf_path):
    # List to store paths of images in the folder
    img_paths = []
    
    # Supported image formats
    supported_formats = ['.jpg', '.jpeg', '.png']
    
    # Loop through all files in the folder
    for file in os.listdir(folder_path):
        # Check if the file is an image
        if os.path.splitext(file)[1].lower() in supported_formats:
            img_paths.append(os.path.join(folder_path, file))
    
    # Sort images by extracting the numerical part of the filename
    img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    
    # Convert images to PDF
    if img_paths:
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert([str(path) for path in img_paths]))
        print(f"PDF created successfully: {output_pdf_path}")
    else:
        print("No supported images found in the folder.")


# Example usage
folder_path = input("Enter the path to the folder containing images: ")
output_pdf_path = input("Enter the path for the output PDF: ")
convert_images_to_pdf(folder_path, output_pdf_path)
