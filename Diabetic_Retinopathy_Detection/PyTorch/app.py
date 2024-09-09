import torch
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50
import time

from flask import Flask, request, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'C:/Users/nikhitha/Desktop/Diabetic_Retinopathy_Detection_R10/PyTorch/'



# Define the model
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



# Initialize the model
device = torch.device("cpu")
model = RetinaModel(num_classes=5).to(device)

loaddict=torch.load('pytorch_model.pt')
model.load_state_dict(loaddict)
model.eval()


@app.route('/')
def home():
    return render_template('home.html')



# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define the endpoint for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    file = request.files['image']
    img = Image.open(file.stream)

    # Preprocess the image
    img_t = transform(img)
    img_t = torch.unsqueeze(img_t, 0)

    # Make a prediction
    with torch.no_grad():
        output = model(img_t)
    probs = torch.softmax(output, dim=1)
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    pred_class = classes[torch.argmax(probs)]

    # Return the prediction result
    result = {'prediction': pred_class}
    return render_template('result.html', text=pred_class)


if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 80, debug=True)