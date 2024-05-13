import rembg
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

class TextRecognitionCNN(nn.Module):
    def __init__(self, num_chars):
        super(TextRecognitionCNN, self).__init__()
        self.features = models.resnet50(pretrained=True) 
        self.features.fc = nn.Linear(2048, 512) 
        self.fc = nn.Linear(512, num_chars) 

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

def preprocess_image(image_path):
    input_image = Image.open(image_path)

    # filtr wyostrzający
    input_image_sharpened = input_image.filter(ImageFilter.UnsharpMask(radius=SHARPEN_RADIUS, percent=150, threshold=3))

    # kontrast
    contrast_enhancer = ImageEnhance.Brightness(input_image_sharpened)
    input_image_enhanced = contrast_enhancer.enhance(2)

    # konwersja do numpy array
    input_array = np.array(input_image_enhanced)

    # usuwanie tła
    output_array = rembg.remove(input_array)

    # konwersja do RGB przed zapisaniem
    output_image_rgb = Image.fromarray(output_array).convert('RGB')

    # normalizacja 
    preprocess = transforms.Compose([
        transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH)),  # Adjust size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(output_image_rgb)


SHARPEN_RADIUS = 1
TARGET_WIDTH = 200
TARGET_HEIGHT = 200
NUM_CHARS = 128  


model = TextRecognitionCNN(NUM_CHARS)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# odczyy danych
data = pd.read_csv('training_data.csv', sep=',', names=['Image_Path', 'Text', 'Is_Correct'])
data = data.dropna(subset=['Text'])

# przetwarzanie obrazów
X = []
y = []
for index, row in data.iterrows():  
    image_path = row['Image_Path']
    text = row['Text']
    is_correct = row['Is_Correct']
    image = preprocess_image(image_path)  
    X.append(image)  
    y.append(1 if is_correct == 'y' else 0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# kowersja danych na tensory pytorch
X_train = torch.stack(X_train).to(device)
X_test = torch.stack(X_test).to(device)

def label_to_one_hot(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

y_train = torch.stack([label_to_one_hot(label, NUM_CHARS) for label in y_train]).to(device)
y_test = torch.stack([label_to_one_hot(label, NUM_CHARS) for label in y_test]).to(device)

# określenie optymalizatora i funkcji straty
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
  
# pętla trenowania i dostosowywanie epok trenowania
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.view(-1, NUM_CHARS), y_train)
    loss.backward()
    optimizer.step()

    # Print training loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# ewaluacja do określenia dokładności
model.eval()
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# ewaluacja precyzji i czułości
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1)
    y_test_argmax = torch.argmax(y_test, dim=1)
    accuracy = torch.mean((predicted == y_test_argmax).float())

    precision = precision_score(y_test_argmax.cpu(), predicted.cpu(), average='macro')
    recall = recall_score(y_test_argmax.cpu(), predicted.cpu(), average='macro')
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Macierz pomyłek
    confusion = confusion_matrix(y_test_argmax.cpu(), predicted.cpu())
    print("Confusion Matrix:")
    print(confusion)


# zapis modelu
torch.save(model.state_dict(), 'text_recognition_model.pth')
print("Model saved.")
