import os
import rembg
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import easyocr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

# Define the Faster CNN model
class FasterCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterCNN, self).__init__()
        self.features = models.resnet50(pretrained=True) # You can choose other pretrained models as well
        self.features.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

# Define preprocess_image function
def preprocess_image(image_path):
    input_image = Image.open(image_path)

    # Sharpening filter
    input_image_sharpened = input_image.filter(ImageFilter.UnsharpMask(radius=SHARPEN_RADIUS, percent=150, threshold=3))

    # Contrast enhancement
    contrast_enhancer = ImageEnhance.Brightness(input_image_sharpened)
    input_image_enhanced = contrast_enhancer.enhance(2)

    # Convert to numpy array
    input_array = np.array(input_image_enhanced)

    # Remove background
    output_array = rembg.remove(input_array)

    # Find text boundaries
    min_row, min_col, max_row, max_col = find_bounding_box(output_array)

    # Crop text area
    output_array_cropped = output_array[min_row:max_row, min_col:max_col]

    # Resize image
    output_image_resized = Image.fromarray(output_array_cropped).resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.BILINEAR)

    # Convert to RGB before saving
    output_image_resized_rgb = output_image_resized.convert('RGB')

    # Apply normalization
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return preprocess(output_image_resized_rgb)

# Define find_bounding_box function
def find_bounding_box(image_array):
    rows = np.any(image_array != [0, 0, 0, 0], axis=(1, 2))
    cols = np.any(image_array != [0, 0, 0, 0], axis=(0, 2))

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return min_row, min_col, max_row, max_col

# Set parameters
SHARPEN_RADIUS = 1
TARGET_WIDTH = 200
TARGET_HEIGHT = 200
NUM_CLASSES = 2  # Binary classification (correct or incorrect)

# Initialize Faster CNN model
model = FasterCNN(NUM_CLASSES)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load data
data = pd.read_csv('training_data.csv', sep=',', names=['Image_Path', 'Text', 'Is_Correct'])
data = data.dropna(subset=['Text'])

# Preprocess images and labels
X = []
y = []
for index, row in data.iterrows():  
    image_path = row['Image_Path']
    text = row['Text']
    is_correct = row['Is_Correct']
    image = preprocess_image(image_path)  # Preprocess image
    X.append(image)
    y.append(1 if is_correct == 'y' else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.stack(X_train).to(device)
X_test = torch.stack(X_test).to(device)
y_train = torch.tensor(y_train).to(device)
y_test = torch.tensor(y_test).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjust learning rate

# Training loop
num_epochs = 30  # Adjust number of epochs
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Print training loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
    print(f"Accuracy: {accuracy}")

# Save the model
torch.save(model.state_dict(), 'faster_cnn_model.pth')
print("Model saved.")

# Continue with prediction as before
