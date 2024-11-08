import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from generator import Generator
from feature_extractor import FeatureExtractor
from utils import load_fer2013_data, preprocess_data, pixel_loss, perceptual_loss, train_expression_direction_model
import os

# Configurations
BATCH_SIZE = 64
LATENT_DIM = 100
LEARNING_RATE = 0.0002
EPOCHS = 100
LAMBDA = 0.5  # Scaling factor for expression direction
OUTPUT_DIR = './outputs/'

# Directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device selection (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
X, y = load_fer2013_data('./data/fer2013.csv')
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Prepare DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models and move to device
feature_extractor = FeatureExtractor().to(device)
generator = Generator(LATENT_DIM).to(device)

# Optimizers
optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

# Train expression direction model (fit the model on CPU or GPU)
latent_vectors = feature_extractor(X_train.to(device))  # Move to device before extracting latent vectors
expression_model = train_expression_direction_model(latent_vectors.detach().cpu().numpy(), y_train.numpy())

# Learning rate decay after 50 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training loop
for epoch in range(EPOCHS):
    generator.train()

    for i, (batch_images, batch_labels) in enumerate(train_loader):
        # Move data to the appropriate device (GPU or CPU)
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        optimizer.zero_grad()

        # Extract latent vectors using feature extractor
        latent_vectors = feature_extractor(batch_images)

        # Adjust latent vectors based on expression model and scaling factor λ
        with torch.no_grad():  # Disable gradients during the expression model prediction
            expression_directions = expression_model.predict(latent_vectors.detach().cpu().numpy())

        adjusted_latent_vectors = latent_vectors + LAMBDA * torch.tensor(expression_directions, dtype=torch.float32).to(device)

        # Generate images
        generated_images = generator(adjusted_latent_vectors)

        # Calculate losses
        loss_pixel = pixel_loss(batch_images, generated_images)
        loss_perceptual = perceptual_loss(batch_images, generated_images, feature_extractor)
        total_loss = loss_pixel + loss_perceptual

        # Backpropagation
        total_loss.backward()
        optimizer.step()

    # Update learning rate
    scheduler.step()

    # Logging and saving checkpoints
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss.item():.4f}')
        torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, f'generator_epoch_{epoch}.pth'))
