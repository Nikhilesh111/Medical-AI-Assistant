# import os
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms # For common image transformations

# # --- Configuration ---
# DATASET_BASE_PATH = "C:\\Users\\YESHAS\\Downloads\\Vector-Database-Application 1\\Vector-Database-Application\\MRI Dataset"
# IMG_SIZE = 128
# NUM_CLASSES = 4 # 'glioma', 'meningioma', 'notumor', 'pituitary'
# BATCH_SIZE = 32
# NUM_EPOCHS = 10 # Start with a small number of epochs for testing

# # Determine if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # --- Custom Dataset Class for PyTorch ---
# class BrainTumorDataset(Dataset):
#     def __init__(self, images, labels, transform=None):
#         self.images = images
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = self.images[idx] # Already a NumPy array (IMG_SIZE, IMG_SIZE)
#         label = self.labels[idx]

#         # Convert NumPy array to PIL Image for torchvision transforms
#         image = Image.fromarray(image)

#         if self.transform:
#             image = self.transform(image)

#         return image, label

# # --- Function to Load Raw Data (same as before) ---
# def load_raw_data(data_dir):
#     images = []
#     labels = []
#     class_names = sorted(os.listdir(data_dir))

#     print(f"Loading raw data from: {data_dir}")
#     print(f"Detected classes: {class_names}")

#     for i, class_name in enumerate(class_names):
#         class_path = os.path.join(data_dir, class_name)
#         if not os.path.isdir(class_path):
#             continue

#         print(f"  Processing class: {class_name} ({i+1}/{len(class_names)})")
#         for image_name in os.listdir(class_path):
#             if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_path = os.path.join(class_path, image_name)
#                 try:
#                     img = Image.open(image_path).convert('L') # Grayscale
#                     img = img.resize((IMG_SIZE, IMG_SIZE))
#                     img_array = np.array(img) # Stored as NumPy array
#                     images.append(img_array)
#                     labels.append(class_name)
#                 except Exception as e:
#                     print(f"    Error loading image {image_path}: {e}")
#     return np.array(images), np.array(labels), class_names

# # --- Define the CNN Model for PyTorch ---
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleCNN, self).__init__()
#         # Input: (Batch, 1, IMG_SIZE, IMG_SIZE) - 1 channel for grayscale
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # Output: (Batch, 32, IMG_SIZE, IMG_SIZE)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (Batch, 32, IMG_SIZE/2, IMG_SIZE/2)

#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # Output: (Batch, 64, IMG_SIZE/2, IMG_SIZE/2)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: (Batch, 64, IMG_SIZE/4, IMG_SIZE/4)

#         # Calculate input features for the fully connected layer
#         # If IMG_SIZE=128, then IMG_SIZE/4 = 32. So 64 * 32 * 32
#         self.fc_input_features = 64 * (IMG_SIZE // 4) * (IMG_SIZE // 4)
#         self.fc = nn.Linear(self.fc_input_features, num_classes)

#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, self.fc_input_features) # Flatten the tensor for the fully connected layer
#         x = self.fc(x)
#         return x

# # --- Main Script ---
# if __name__ == "__main__":
#     # 1. Load Raw Image and Label Data
#     train_raw_images, train_raw_labels, class_names = load_raw_data(os.path.join(DATASET_BASE_PATH, 'Training'))
#     test_raw_images, test_raw_labels, _ = load_raw_data(os.path.join(DATASET_BASE_PATH, 'Testing'))

#     print("\n--- Data Loading Summary ---")
#     print(f"Training images raw shape: {train_raw_images.shape}")
#     print(f"Training labels raw shape: {train_raw_labels.shape}")
#     print(f"Testing images raw shape: {test_raw_images.shape}")
#     print(f"Testing labels raw shape: {test_raw_labels.shape}")
#     print(f"Number of classes: {len(class_names)}")
#     print(f"Class names: {class_names}")

#     # 2. Encode Labels
#     label_encoder = LabelEncoder()
#     train_labels_encoded = label_encoder.fit_transform(train_raw_labels)
#     test_labels_encoded = label_encoder.transform(test_raw_labels)

#     print("\n--- Label Encoding Example ---")
#     print(f"Original label (first training image): {train_raw_labels[0]}")
#     print(f"Encoded label (first training image): {train_labels_encoded[0]}")

#     # 3. Define Image Transformations for PyTorch
#     # Normalization for PyTorch: images are typically 0-1, then normalized by mean/std
#     # For grayscale, mean/std can be approx 0.5/0.5 or dataset-specific
#     transform = transforms.Compose([
#         # Image.fromarray(numpy_array) is done in __getitem__
#         transforms.ToTensor(), # Converts PIL Image (H, W) or NumPy (H, W, C) to PyTorch Tensor (C, H, W) and scales to [0, 1]
#         transforms.Normalize(mean=[0.5], std=[0.5]) # For grayscale images (1 channel)
#     ])

#     # 4. Create PyTorch Datasets
#     # Split training data into training and validation sets
#     X_train_data, X_val_data, y_train_labels, y_val_labels = train_test_split(
#         train_raw_images, train_labels_encoded, test_size=0.2, random_state=42, stratify=train_labels_encoded
#     )

#     train_dataset = BrainTumorDataset(X_train_data, y_train_labels, transform=transform)
#     val_dataset = BrainTumorDataset(X_val_data, y_val_labels, transform=transform)
#     test_dataset = BrainTumorDataset(test_raw_images, test_labels_encoded, transform=transform)

#     # 5. Create PyTorch DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#     print("\n--- DataLoader Information ---")
#     print(f"Number of training batches: {len(train_loader)}")
#     print(f"Number of validation batches: {len(val_loader)}")
#     print(f"Number of testing batches: {len(test_loader)}")

#     # 6. Initialize Model, Loss, and Optimizer
#     model = SimpleCNN(NUM_CLASSES).to(device) # Move model to GPU if available
#     criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     print("\n--- Starting Model Training ---")
#     # 7. Training Loop
#     for epoch in range(NUM_EPOCHS):
#         model.train() # Set model to training mode
#         running_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         for i, (inputs, labels) in enumerate(train_loader):
#             inputs = inputs.to(device) # Move inputs to GPU
#             labels = labels.to(device) # Move labels to GPU

#             optimizer.zero_grad() # Zero the parameter gradients
#             outputs = model(inputs) # Forward pass
#             loss = criterion(outputs, labels) # Calculate loss
#             loss.backward() # Backward pass
#             optimizer.step() # Optimize

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total_train += labels.size(0)
#             correct_train += (predicted == labels).sum().item()

#         epoch_loss = running_loss / len(train_dataset)
#         epoch_acc = correct_train / total_train

#         # --- Validation Loop ---
#         model.eval() # Set model to evaluation mode
#         val_loss = 0.0
#         correct_val = 0
#         total_val = 0
#         with torch.no_grad(): # Disable gradient calculation for validation
#             for inputs_val, labels_val in val_loader:
#                 inputs_val = inputs_val.to(device)
#                 labels_val = labels_val.to(device)
#                 outputs_val = model(inputs_val)
#                 loss_val = criterion(outputs_val, labels_val)
#                 val_loss += loss_val.item() * inputs_val.size(0)
#                 _, predicted_val = torch.max(outputs_val.data, 1)
#                 total_val += labels_val.size(0)
#                 correct_val += (predicted_val == labels_val).sum().item()

#         val_epoch_loss = val_loss / len(val_dataset)
#         val_epoch_acc = correct_val / total_val

#         print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

#     print("\n--- Training Complete! ---")

#     # 8. Evaluate Model on Test Set
#     model.eval() # Set model to evaluation mode
#     correct_test = 0
#     total_test = 0
#     with torch.no_grad():
#         for inputs_test, labels_test in test_loader:
#             inputs_test = inputs_test.to(device)
#             labels_test = labels_test.to(device)
#             outputs_test = model(inputs_test)
#             _, predicted_test = torch.max(outputs_test.data, 1)
#             total_test += labels_test.size(0)
#             correct_test += (predicted_test == labels_test).sum().item()

#     test_accuracy = correct_test / total_test
#     print(f"\nTest Accuracy: {test_accuracy:.4f}")

#     # 9. Save the trained model (optional)
#     torch.save(model.state_dict(), 'brain_tumor_model.pth')
#     print("Model saved to brain_tumor_model.pth")