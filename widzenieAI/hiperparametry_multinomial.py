
import rembg
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import easyocr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


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

    return np.array(output_image_resized_rgb)  

# promień wyostrzania
SHARPEN_RADIUS = 1
TARGET_WIDTH = 200
TARGET_HEIGHT = 200

# inicjalizacja easyocr
reader = easyocr.Reader(['en'])

training_data_path = 'training_data.csv'

data = pd.read_csv('training_data.csv', sep=',', names=['Image_Path', 'Text', 'Is_Correct'])

print(data.head())

X = []
y = []
for index, row in data.iterrows():  
    image_path = row['Image_Path']
    text = row['Text']
    is_correct = row['Is_Correct']
    image = preprocess_image(image_path) 
    X.append(np.array(image)) 
    y.append(1 if is_correct == 'y' else 0)

print("INFO: przed konwersją obrazów na tablicę numpy")
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ekstrakcja cech
vectorizer = CountVectorizer()
# usuwanie wierszy zawierające wartości NaN
data = data.dropna(subset=['Text'])

# przekształcenie tekstu na cechy
X_train_vec = vectorizer.fit_transform(data['Text'])

# modyfikacja hiperparametrów
from sklearn.model_selection import GridSearchCV

hyperparameters = {'alpha': [1, 0.5, 1.0],
                   'fit_prior': [True, False]}

grid_search = GridSearchCV(MultinomialNB(), hyperparameters, cv=5, verbose=0)
grid_search.fit(X_train_vec, y_train)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)


best_classifier = MultinomialNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])
best_classifier.fit(X_train_vec, y_train)

# ocena modelu na danych testowych
X_test_vec = vectorizer.transform(data['Text'])
y_pred = best_classifier.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred[:len(y_test)])
print("Dokładność modelu: {:.2f}%".format(accuracy * 100))

precision = precision_score(y_test, y_pred[:len(y_test)])
print("Precyzja modelu: {:.2f}".format(precision))

recall = recall_score(y_test, y_pred[:len(y_test)])
print("Czułość modelu: {:.2f}".format(recall))

conf_matrix = confusion_matrix(y_test, y_pred[:len(y_test)])
print("Macierz pomyłek:")
print(conf_matrix)