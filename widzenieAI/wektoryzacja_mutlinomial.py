import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, accuracy_score, precision_score, recall_score, confusion_matrix
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

# wczytanie danych
data = pd.read_csv('training_data.csv', sep=',', names=['Image_Path', 'Text', 'Is_Correct'])

# przetwarzanie obrazów
X = []
y = []
for index, row in data.iterrows():  
    image_path = row['Image_Path']
    text = row['Text']
    is_correct = row['Is_Correct']
    image = preprocess_image(image_path)  
    X.append(np.array(image))  
    y.append(1 if is_correct == 'y' else 0)

X = np.array(X)


X_train_img, X_test_img, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


data = data.dropna(subset=['Text'])

# wektoryzacja tekstu
tfidf_vectorizer = TfidfVectorizer()
X_train_text_tfidf = tfidf_vectorizer.fit_transform(data['Text'])

X_test_text_tfidf = tfidf_vectorizer.transform(data['Text'])

classifier = MultinomialNB()
classifier.fit(X_train_text_tfidf, y_train)

y_pred = classifier.predict(X_test_text_tfidf)

accuracy = accuracy_score(y_test, y_pred[:len(y_test)])
print("Dokładność modelu: {:.2f}%".format(accuracy * 100))

precision = precision_score(y_test, y_pred[:len(y_test)])
print("Precyzja modelu: {:.2f}".format(precision))

recall = recall_score(y_test, y_pred[:len(y_test)])
print("Czułość modelu: {:.2f}".format(recall))

conf_matrix = confusion_matrix(y_test, y_pred[:len(y_test)])
print("Macierz pomyłek:")
print(conf_matrix)