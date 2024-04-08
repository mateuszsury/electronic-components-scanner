import os
import rembg
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import easyocr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import csv
import os

folder_path = r'C:\Users\mklim\Desktop\Informatyczne systemy automatyki\electronic-components-scanner-main\ai_jpg'

def find_bounding_box(image_array):
    rows = np.any(image_array != [0, 0, 0, 0], axis=(1, 2))
    cols = np.any(image_array != [0, 0, 0, 0], axis=(0, 2))

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return min_row, min_col, max_row, max_col

def preprocess_image(image_path):
    input_image = Image.open(image_path)

    # filtr wyostrzający z dostosowanym promieniem
    input_image_sharpened = input_image.filter(ImageFilter.UnsharpMask(radius=SHARPEN_RADIUS, percent=150, threshold=3))

    # kontrast
    contrast_enhancer = ImageEnhance.Brightness(input_image_sharpened)
    input_image_enhanced = contrast_enhancer.enhance(2)

    # konwersja do numpy array
    input_array = np.array(input_image_enhanced)

    # usuwanie tła
    output_array = rembg.remove(input_array)

    # znajdź granice tekstu
    min_row, min_col, max_row, max_col = find_bounding_box(output_array)

    # wytnij obszar z tekstem
    output_array_cropped = output_array[min_row:max_row, min_col:max_col]

    # przeskaluj obraz
    output_image_resized = Image.fromarray(output_array_cropped).resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.BILINEAR)

    # konwersja do RGB przed zapisaniem
    output_image_resized_rgb = output_image_resized.convert('RGB')

    return np.array(output_image_resized_rgb)  # Zwróć obraz jako tablicę numpy

    

# promień wyostrzania
SHARPEN_RADIUS = 1
TARGET_WIDTH = 200
TARGET_HEIGHT = 200

# inicjalizacja easyocr
reader = easyocr.Reader(['en'])

training_data_path = 'training_data.csv'

import pandas as pd

data = pd.read_csv('training_data.csv', sep=',', names=['Image_Path', 'Text', 'Is_Correct'])

print(data.head())

X = []
y = []
for index, row in data.iterrows():  
    image_path = row['Image_Path']
    text = row['Text']
    is_correct = row['Is_Correct']
    image = preprocess_image(image_path)  # przetworzenie obrazu
    X.append(np.array(image))  # dodanie przetworzonego obrazu do listy X
    y.append(1 if is_correct == 'y' else 0)

print("przed konwersją obrazów na tablicę numpy")
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ekstrakcja cech
vectorizer = CountVectorizer()
# usuwanie wierszy zawierające wartości NaN
data = data.dropna(subset=['Text'])

# przekształcenie tekstu na cechy
X_train_vec = vectorizer.fit_transform(data['Text'])

classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# ocena modelu na danych testowych
X_test_vec = vectorizer.transform(data['Text'])
y_pred = classifier.predict(X_test_vec)

# dokładność
accuracy = accuracy_score(y_test, y_pred[:len(y_test)])
print(f"Accuracy: {accuracy}")

test_folder_path = r'C:\Users\mklim\Desktop\Informatyczne systemy automatyki\electronic-components-scanner-main\ai_jpg'
test_images = [os.path.join(test_folder_path, filename) for filename in os.listdir(test_folder_path)]

# predykcja
results = []
for image_path in test_images:
    image = preprocess_image(image_path)
    text = ' '.join([res[1] for res in reader.readtext(np.array(image))])
    text_vec = vectorizer.transform([text])
    prediction = classifier.predict(text_vec)
    results.append((image_path, text, prediction[0]))

# zapis do pliku
output_csv_path = 'predictions.csv'
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image_Path', 'Text', 'Prediction'])
    writer.writerows(results)
    
print("Results saved to", output_csv_path)