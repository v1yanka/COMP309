import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.io import read_image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import os
import random
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

SEED = 309
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

base_dir = os.getcwd()
test_dir = os.path.join(base_dir,'/content/drive/MyDrive/Colab Notebooks/code/testdata')
class_names = ['cherry', 'strawberry', 'tomato']

def image_size_filter(path):
    try:
        with Image.open(path) as img:
            # Convert to RGB if not already in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return True  # Indicate that the file is valid
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return False

test_transforms = transforms.Compose([
    transforms.Resize((75, 75)),  # resize for consistency and faster runtime
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testdata = datasets.ImageFolder(root=test_dir, transform=test_transforms,
    is_valid_file=image_size_filter  # applying the filter
  )

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

# define the CNN model architecture (match the architecture used in training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 9 * 9)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the saved model
######################## UPDATE PATH IN ECS COMPUTER
model = CNNModel().to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/code/model.pth', map_location=device))
model.eval()  # Set to evaluation mode for inference


# load test data
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

# class names (must match the order of classes in training)
class_names = ('cherry', 'strawberry', 'tomato')

# run inference on test set
def evaluate_model(model, test_loader):
    model.eval()  # ensure dropout/batchnorm layers are in evaluation mode
    predictions = []

    with torch.no_grad():  # no gradient computation for inference
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max logit
            predictions.append(predicted.item())

    return predictions

# get predictions
predictions = evaluate_model(model, test_loader)

# print predictions for each test image
#for i, pred in enumerate(predictions):
#    print(f"Image {i+1}: Predicted class - {class_names[pred]}")

# code to get accuracy of the model

# create a list to store true labels and predicted labels
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# calculate accuracy using sklearn's accuracy_score
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")