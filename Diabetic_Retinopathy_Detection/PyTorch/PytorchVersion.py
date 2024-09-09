import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50
import time
# Read dataset
dataset = pd.read_csv('C:/Users/nikhitha/Desktop/Dataset/trainLabels.csv', on_bad_lines='skip')
image_folder_path = "C:/Users/nikhitha/Desktop/Dataset/Images"

labels = pd.Series(dataset['level']).to_numpy()

x = []
y = []
fixed_size = (300, 300)

# resizes images to a fixed size of 300x300
for index, row in dataset.iterrows():
    image_path = os.path.join(image_folder_path, row['image']+'.jpeg')
    image = cv2.imread(image_path)
    if image is None:
        pass
    else:
        resized_image = cv2.resize(image, fixed_size)
        x.append(resized_image)
        y.append(labels[index])

# stores them in a NumPy array along with their corresponding labels.
X = np.concatenate((x, x), axis=0)
Y = np.concatenate((y, y), axis=0)
del x
del y
del labels

#Splits the data into training and validation sets using a 80:20 ratio.
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=42)

# Custom dataset class that takes in the images and labels and applies a set of transformations to them.
class RetinaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Defines a RetinaNet model class that uses the ResNet50 backbone and adds some fully connected layers to output the final classification.
class RetinaModel(nn.Module):
    def __init__(self, num_classes):
        super(RetinaModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = RetinaDataset(x_train, y_train, transform=transform)
valid_dataset = RetinaDataset(x_valid, y_valid, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Initialize the model
device = torch.device("cpu")
model = RetinaModel(num_classes=5).to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 15
best_valid_loss = float('inf')
early_stopping_counter = 0

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

#Defines a training loop that trains the model for a specified number of epochs, evaluating the model on the validation set at each epoch and saving the best model.
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_corrects = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)

    model.eval()
    valid_loss = 0.0
    valid_corrects = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            valid_loss += loss.item() * inputs.size(0)
            valid_corrects += torch.sum(preds == labels.data)

    epoch_train_loss = train_loss / len(train_dataset)
    epoch_valid_loss = valid_loss / len(valid_dataset)
    epoch_train_acc = train_corrects.double() / len(train_dataset)
    epoch_valid_acc = valid_corrects.double() / len(valid_dataset)
    
    
    train_accuracies.append(epoch_train_acc.item())
    val_accuracies.append(epoch_valid_acc.item())
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_valid_loss)
    # Prints out the training and validation losses and accuracies at each epoch 
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"train loss: {epoch_train_loss:.4f}, train acc: {epoch_train_acc:.4f}, "
          f"Test loss: {epoch_valid_loss:.4f}, Test acc: {epoch_valid_acc:.4f}")

    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss
        torch.save(model.state_dict(), 'pytorch_model.pt')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= 15:
            print("Early stopping")
            break



#Prints out the total training time.          
end_time = time.time()
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))         
final_val_acc_pt = val_accuracies[-1]
print("Final validation accuracy for PyTorch: {:.4f}".format(final_val_acc_pt))
import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)

plt.plot(epochs, train_accuracies, label='Training')
plt.plot(epochs, val_accuracies, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and validation accuracy')
plt.show()

plt.plot(epochs, train_losses, label='Training')
plt.plot(epochs, val_losses, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and validation loss')
plt.show()