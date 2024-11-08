# training/utils.py

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LinearRegression
import cv2
from sklearn.model_selection import train_test_split

# Load FER2013 dataset
def load_fer2013_data(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    for _, row in df.iterrows():
        image = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        images.append(image)
        labels.append(row['emotion'])
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def preprocess_data(images, labels, test_size=0.2):
    images = images / 255.0  # Normalize images
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    X_train, X_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), test_size=test_size, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test

# Pixel loss
def pixel_loss(real_image, generated_image):
    return torch.mean(torch.abs(real_image - generated_image))

# Perceptual loss
def perceptual_loss(real_image, generated_image, feature_extractor):
    real_features = feature_extractor(real_image)
    generated_features = feature_extractor(generated_image)
    return torch.mean((real_features - generated_features) ** 2)

# Train expression direction model (Linear Regression)
def train_expression_direction_model(latent_vectors, expression_labels):
    model = LinearRegression()
    model.fit(latent_vectors, expression_labels)
    return model
