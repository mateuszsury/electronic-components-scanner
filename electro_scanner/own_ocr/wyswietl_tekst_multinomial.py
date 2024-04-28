import joblib
import numpy as np
from PIL import Image
import easyocr
import rembg
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance  # Dodany import

def preprocess_image(image_path):
    input_image = Image.open(image_path)
    
    SHARPEN_RADIUS = 1
    
    input_image_sharpened = input_image.filter(ImageFilter.UnsharpMask(radius=SHARPEN_RADIUS, percent=150, threshold=3))

    contrast_enhancer = ImageEnhance.Brightness(input_image_sharpened)
    input_image_enhanced = contrast_enhancer.enhance(2)

    input_array = np.array(input_image_enhanced)

    output_array = rembg.remove(input_array)

    rows = np.any(output_array != [0, 0, 0, 0], axis=(1, 2))
    cols = np.any(output_array != [0, 0, 0, 0], axis=(0, 2))

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    output_array_cropped = output_array[min_row:max_row, min_col:max_col]

    output_image_resized = Image.fromarray(output_array_cropped)

    output_image_resized_rgb = output_image_resized.convert('RGB')

    return output_image_resized_rgb


def load_model_and_vectorizer(model_and_vectorizer_path):
    # wczytanie modelu i wektoryzatora
    loaded_model, loaded_vectorizer = joblib.load(model_and_vectorizer_path)
    return loaded_model, loaded_vectorizer

def read_text_from_image(image_path, model_and_vectorizer_path):

    classifier, vectorizer = load_model_and_vectorizer(model_and_vectorizer_path)
    
    processed_image = preprocess_image(image_path)
    
    reader = easyocr.Reader(['en'])
    
    text = ' '.join([res[1] for res in reader.readtext(np.array(processed_image))])
    
    # przekształcenie tekstu na wektor
    text_vec = vectorizer.transform([text])
    print(text_vec)
    
    prediction = classifier.predict(text_vec)
    
    return text, prediction


