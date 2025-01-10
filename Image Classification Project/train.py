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

# make sure code runtime type is "Python 3 Google Compute Engine backend (GPU)"
# so gpu can be used

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

SEED = 309
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



# using my current working directory as base directory
base_dir = os.getcwd()

# joining base_dir with relative path to train_data
data_dir = os.path.join(base_dir, '/content/drive/MyDrive/Colab Notebooks/code/train_data')
# only including files with dimensions 300x300
def image_size_filter(path):
    try:
        with Image.open(path) as img:
            # Convert to RGB if not already in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Check for size after conversion
            return img.size == (300, 300)
    except IOError: # for errors like opening non-image files
        return False # using those errors to exclude file≠images

# the transformations and augments
train_transforms = transforms.Compose([
    transforms.Resize((75, 75)),  # resize for consistency and faster runtime
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(p=0.5),  #random horizontal flip w prob 0.5
    transforms.RandomRotation(15), #random rotations
])


# get the dataset with flexible path
train_data = datasets.ImageFolder(
    root=data_dir,
    transform=train_transforms,
    is_valid_file=image_size_filter  # applying the filter
)
# Confirm data loading works
print("Classes:", train_data.classes)
print("Number of samples:", len(train_data))

"""- base_dir = os.getcwd()
- test_dir = os.path.join(base_dir, '/content/drive/MyDrive/Colab Notebooks/code/testdata')
- def image_size_filter(path):
    try:
        with Image.open(path) as img:
            # Convert to RGB if not already in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Check for size after conversion
            return img.size == (300, 300)
    except IOError: # for errors like opening non-image files
        return False # using those errors to exclude file≠images
- test_transforms = transforms.Compose([
    transforms.Resize((75, 75)),  # resize for consistency and faster runtime
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
- test_data = datasets.ImageFolder(
    root=test_dir,
    transform=test_transforms,
    is_valid_file=image_size_filter  # applying the filter
)
"""

val_ratio = 0.8 # defining the train to validation ratio for split

train_props = int(len(train_data) * val_ratio)
val_props = len(train_data) - train_props

train_ds, val_ds = torch.utils.data.random_split(train_data, [train_props, val_props])
batch_size = 64

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                         num_workers=2)
classes = ('cherry','strawberry','tomato')

class_idxs = train_data.class_to_idx

# getting class counts
total_counts = {class_label: 0 for class_label in class_idxs}

# Count the occurrences of each class label
for _, label in train_ds:
    total_counts[train_data.classes[label]] += 1

# Print the counts
for class_label, count in total_counts.items():
    print(f'Class: {class_label}: {count} train entries')

total_counts = {class_label: 0 for class_label in class_idxs}
for _, label in val_ds:
    total_counts[train_data.classes[label]] += 1

# Print the counts
for class_label, count in total_counts.items():
    print(f'Class: {class_label}: {count} val entries')

"""## Baseline MLP

"""

class MLPBaseline(nn.Module):
    def __init__(self, input_size=75*75*3, hidden_dim=512, output_dim=3):
        super(MLPBaseline, self).__init__()

        # first fully connected layer with input size matching flattened image
        # dimensions since i resized the image is resized to 75x75 with 3 color
        # channels, so input size = 75*75*3
        self.fc1 = nn.Linear(input_size, hidden_dim)

        # relu activ funct for non-linearity
        # relu helps the model learn more complex patterns by applying
        # non-linearity
        self.relu = nn.ReLU()

        # second fully connected layer, output size = number of classes (3)
        # the final layer has 3 outputs, one for each class (cherry, strawberry,
        # tomato)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # flatten input from (batch_size, 3, 75, 75) to (batch_size, 75*75*3)
        x = x.view(-1, 75*75*3)

        # pass through first fully connected layer
        x = self.fc1(x)

        # appling relu activ
        x = self.relu(x)

        # second pass thru fully connected layer to get class scores
        x = self.fc2(x)
        return x  # output the logits for each class

# initialising mlp base model, set loss function, and optimiser
mlp_model = MLPBaseline().to(device)  # moving model to gpu if available
criterion = nn.CrossEntropyLoss()      # cross-entropy loss for multi-class classification
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)  # adam optimiser w default learning rate

# training function for the MLP model
def train_mlp(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    model.train()  # set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0  # track cumulative loss for the epoch

        # iterate over batches of data in the training loader
        for images, labels in train_loader:
            # move images and labels to device (gpu if available)
            images, labels = images.to(device), labels.to(device)

            # reset grads
            optimizer.zero_grad()

            # forward pass: compute model outputs
            outputs = model(images)

            # compute loss between model outputs and true labels
            loss = criterion(outputs, labels)

            # backward pass: compute gradients
            loss.backward()

            # update model parameters
            optimizer.step()

            # accumulate loss for monitoring
            running_loss += loss.item()

        # Calculate and print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validate model on validation data after each epoch
        validate_mlp(model, val_loader)

# Function to evaluate model performance on validation set
def validate_mlp(model, val_loader):
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    val_loss = 0.0  # Track validation loss
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculation for validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            val_loss += loss.item()  # Accumulate loss

            # Convert outputs to predicted classes
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_preds.extend(preds.cpu().numpy())    # Store predictions

    # Calculate and print validation accuracy
    val_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
    model.train()  # Set model back to training mode

# Train the MLP model
train_mlp(mlp_model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

"""# Building CNN Model

"""

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # first conv layer: in channels = 3 (RGB), out channels = 32
        # this learns 32 filters of size 3x3 to capture basic patterns
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # Batch normalization after conv1 to stabilize training
        self.bn1 = nn.BatchNorm2d(32)

        # second conv layer: in chnls = 32, out chnls = 64
        # learns 64 filters, captures more complex patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # third conv layer: in chnls = 64, out chnls = 128
        # learns even more complex feats with 128 filters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # max pooling layer with kernel size 2x2 and stride 2
        # reduces spatial dims by half so we can get most prominent feats
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layer after flattening the output from conv layers
        # flattened size will be 128 * 9 * 9 due to pooling (w input 75x75)
        self.fc1 = nn.Linear(128 * 9 * 9, 512)

        # output layer w num of classes for the classification
        self.fc2 = nn.Linear(512, 3)

        # activation and dropout for regularisation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # con layer 1 + batchnorm + relu + pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        # con layer 2 + batchnorm + relu + pooling
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # con layer 3 + batchnorm + relu + pooling
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # flattengin output from con layers
        x = x.view(-1, 128 * 9 * 9)

        # fullu connected layer 1 + relu + dropout
        x = self.dropout(self.relu(self.fc1(x)))

        # output layer (logits for each class)
        x = self.fc2(x)

        return x

# setting up cnn model, loss function, and optimiser
cnn_model = CNNModel().to(device)  # move model to gpu if available
cnn_criterion = nn.CrossEntropyLoss()  # cross ent loss for multi-class
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)  # Adam optimiser

# check the cnn model architecture
print(cnn_model)

train_mlp(cnn_model, train2_loader, val2_loader, label_smoothing_criterion, cnn_optimizer_adamw, num_epochs=10)

"""## cross validation"""

from sklearn.model_selection import KFold
from torch.utils.data import Subset

def cross_validation_training(model_class, train_data, num_folds=5, num_epochs=10):
 kfold = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
 fold_results = []

 for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
     print(f"Fold {fold+1}")

     # train and val subsets for this fold
     train_sub = Subset(train_data, train_idx)
     val_sub = Subset(train_data, val_idx)

     # data loaders
     train_loader = DataLoader(train_sub, batch_size=64, shuffle=True, num_workers=2)
     val_loader = DataLoader(val_sub, batch_size=64, num_workers=2)

     # set up model, criterion, and optimizer for each fold
     model = model_class().to(device)
     criterion = nn.CrossEntropyLoss()
     optimizer = optim.Adam(model.parameters(), lr=0.001)

     # train and validate
     train_mlp(model, train_loader, val_loader, criterion, optimizer, num_epochs)
     fold_results.append(validate_mlp(model, val_loader))  # Collect validation accuracy for each fold

 print("Cross-validation results:", fold_results)

# now running cross val on cnnmodel
cross_validation_training(CNNModel, train_data, num_folds=5, num_epochs=10)

"""## label smoothing"""

class LabelSmoothingLoss(nn.Module):
 def __init__(self, classes=3, smoothing=0.1):
     super(LabelSmoothingLoss, self).__init__()
     self.confidence = 1.0 - smoothing
     self.smoothing = smoothing
     self.cls = classes

 def forward(self, x, target):
     log_probs = F.log_softmax(x, dim=-1)
     targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), self.confidence)
     targets += self.smoothing / (self.cls - 1)
     loss = (-targets * log_probs).mean()
     return loss

# using label smoothing loss
label_smoothing_criterion = LabelSmoothingLoss(classes=3, smoothing=0.1).to(device)
train_mlp(cnn_model, train_loader, val_loader, label_smoothing_criterion, cnn_optimizer, num_epochs=10)

"""## changing optimisers"""

# sgd with momentum
cnn_optimizer_sgd = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
train_mlp(cnn_model, train_loader, val_loader, label_smoothing_criterion, cnn_optimizer_sgd, num_epochs=10)

# adamw optimiser
cnn_optimizer_adamw = optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=1e-4)
train_mlp(cnn_model, train_loader, val_loader, label_smoothing_criterion, cnn_optimizer_adamw, num_epochs=10)

# update the transformations to add color jitter and random crop
train2_transforms = transforms.Compose([
 transforms.Resize((75, 75)),
 transforms.RandomHorizontalFlip(p=0.5),
 transforms.RandomRotation(15),
 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
 transforms.RandomCrop((75, 75), padding=4),
 transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# re-apply transformations to dataset
train2_data = datasets.ImageFolder(
 root=data_dir,
 transform=train2_transforms,
 is_valid_file=image_size_filter
)

# reload train loader with updated dataset
train2_loader = DataLoader(train2_data, batch_size=batch_size, shuffle=True, num_workers=2)
val2_loader = DataLoader(train2_data, batch_size=batch_size, shuffle=True, num_workers=2)

# train the model with augmented data
train_mlp(cnn_model, train_loader, val_loader, label_smoothing_criterion, cnn_optimizer, num_epochs=10)



train_ds2, val_ds2 = torch.utils.data.random_split(train2_data, [train_props, val_props])
batch_size = 64

train2_loader = torch.utils.data.DataLoader(train_ds2, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
val2_loader = torch.utils.data.DataLoader(val_ds2, batch_size=batch_size,
                                         num_workers=2)
classes = ('cherry','strawberry','tomato')
# Train the model with augmented data
train_mlp(cnn_model, train2_loader, val2_loader, label_smoothing_criterion, cnn_optimizer_adamw, num_epochs=10)

torch.save(cnn_model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/code/model.pth')

mlp_val_accuracy = [0.4611, 0.5017, 0.5051, 0.5220, 0.5276, 0.5287, 0.5152, 0.5242, 0.5152, 0.5141]
mlp_train_loss = [1.9831, 1.1609, 1.0933, 1.0125, 0.9823, 0.9783, 0.9606, 0.9210, 0.9156, 0.9513]
mlp_val_loss = [1.2549, 1.2407, 1.0851, 1.0175, 1.0611, 1.0214, 1.0420, 1.0699, 1.0662, 1.1318]

cnn_val_accuracy = [0.7486, 0.7441, 0.8061, 0.8185, 0.7813, 0.7790, 0.7463, 0.7328, 0.8095, 0.8162]
cnn_train_loss = [0.2670, 0.2624, 0.2591, 0.2600, 0.2608, 0.2630, 0.2585, 0.2617, 0.2540, 0.2570]
cnn_val_loss = [0.6327, 0.6300, 0.5600, 0.5383, 0.5697, 0.5863, 0.6169, 0.6370, 0.5339, 0.5660]



# Create plots for validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(mlp_val_accuracy, label='MLP')
plt.plot(cnn_val_accuracy, label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.show()

# Create plots for training loss
plt.figure(figsize=(10, 5))
plt.plot(mlp_train_loss, label='MLP')
plt.plot(cnn_train_loss, label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()


# Create plots for validation loss
plt.figure(figsize=(10, 5))
plt.plot(mlp_val_loss, label='MLP')
plt.plot(cnn_val_loss, label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.show()

"""# trying pretrained models"""



