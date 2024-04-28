import os
import rembg
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def find_bounding_box(image_array):
    rows = np.any(image_array != [0, 0, 0, 0], axis=(1, 2))
    cols = np.any(image_array != [0, 0, 0, 0], axis=(0, 2))

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return min_row, min_col, max_row, max_col
def process_image(image_path):
    TARGET_WIDTH = 200
    TARGET_HEIGHT = 200

    # Dostosuj promień wyostrzania
    SHARPEN_RADIUS = 15

    # Poproś użytkownika o ścieżkę do obrazu

    input_image = Image.open(image_path)

    # Zastosuj filtr wyostrzający z dostosowanym promieniem
    input_image_sharpened = input_image.filter(ImageFilter.UnsharpMask(radius=SHARPEN_RADIUS, percent=150, threshold=3))

    contrast_enhancer = ImageEnhance.Brightness(input_image_sharpened)
    input_image_enhanced = contrast_enhancer.enhance(2)

    input_array = np.array(input_image_enhanced)
    output_array = rembg.remove(input_array)

    min_row, min_col, max_row, max_col = find_bounding_box(output_array)
    output_array_cropped = output_array[min_row:max_row, min_col:max_col]

    output_image = Image.fromarray(output_array_cropped)

    # Konwertuj obraz do trybu RGB przed zapisaniem jako JPEG
    output_image_resized_rgb = output_image.convert('RGB')

    output_image_resized_rgb.save('processed.jpg', quality=100)  # Dostosuj jakość do poziomu kompresji, np. quality=90
    print("Image processed")
    return 'processed.jpg'


