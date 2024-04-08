# w programie następuje sformatowanie zdjęcia, jego zapis i odczyt napisu przy pomocy easyocr; następnie weryfikowany manualnie jest odczyt i zapisane informacje do .csv
import os
import rembg
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import easyocr

def find_bounding_box(image_array):
    rows = np.any(image_array != [0, 0, 0, 0], axis=(1, 2))
    cols = np.any(image_array != [0, 0, 0, 0], axis=(0, 2))

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return min_row, min_col, max_row, max_col

TARGET_WIDTH = 200
TARGET_HEIGHT = 200
folder_path = r'C:\Users\mklim\Desktop\Informatyczne systemy automatyki\electronic-components-scanner-main\example_img\baza_obrazow_jpg'

# promień wyostrzania
SHARPEN_RADIUS = 1

# inicjalizacja EasyOCR 
reader = easyocr.Reader(['en'])

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        input_image = Image.open(image_path)

        # filtr wyostrzający z dostosowanym promieniem
        input_image_sharpened = input_image.filter(ImageFilter.UnsharpMask(radius=SHARPEN_RADIUS, percent=150, threshold=3))

        # kontrast
        contrast_enhancer = ImageEnhance.Brightness(input_image_sharpened)
        input_image_enhanced = contrast_enhancer.enhance(2)

        input_array = np.array(input_image_enhanced)
        
        # usuwanie tła
        output_array = rembg.remove(input_array)

        # znajdź granice tekstu
        min_row, min_col, max_row, max_col = find_bounding_box(output_array)

        # wytnij obszar z tekstem
        output_array_cropped = output_array[min_row:max_row, min_col:max_col]

        # tworzenie obrazu z numpy array
        output_image = Image.fromarray(output_array_cropped)

        # przeskaluj obraz
        output_image_resized = output_image.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.BILINEAR)

        # konwersja do RGB przed zapisaniem
        output_image_resized_rgb = output_image_resized.convert('RGB')

        output_path = os.path.join(r'C:\Users\mklim\Desktop\Informatyczne systemy automatyki\electronic-components-scanner-main\example_img_resized', f'{os.path.splitext(filename)[0]}_resized.jpg')
        output_image_resized_rgb.save(output_path, quality=90)  # dostosuj jakość do poziomu kompresji, np. quality=90

        # odczytaj tekst z obrazu za pomocą EasyOCR
        result = reader.readtext(output_path)
        text = ' '.join([res[1] for res in result])
        print(f"Napisy z obrazu {filename}: {text}")

        # manualne wprowadzenie informacji o poprawności odczytanego tekstu
        is_correct = input("Czy tekst został poprawnie odczytany? (y/n): ").strip().lower()
        while is_correct not in {'y', 'n'}:
            print("Niepoprawny wybór. Proszę wpisać 'y' lub 'n'.")
            is_correct = input("Czy tekst został poprawnie odczytany? (y/n): ").strip().lower()

        # zapisanie informacji do pliku csv
        with open('training_data.csv', 'a') as file:
            file.write(f"{output_path},{text},{is_correct}\n")

print("Przetworzono wszystkie obrazy.")
