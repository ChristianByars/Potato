import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import collections.abc
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

collections.Iterable = collections.abc.Iterable


#number of classifications and input shape
num_classes = 10
input_size = 3*224*224
input_channels = 3

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor()
    ]
)

train_path = r"C:\Users\chris\OneDrive\Desktop\SDSU\Pavement Distress\Potato\FinalDataset\train"
test_path = r"C:\Users\chris\OneDrive\Desktop\SDSU\Pavement Distress\Potato\FinalDataset\test"

train_dataset = datasets.ImageFolder(root = train_path, transform=transform)
test_dataset = datasets.ImageFolder(root = test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size= 8, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size= 8, shuffle = True)

#image shape: [3,224,224]
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #stage 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
         #stage 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
         #stage 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        #connected layer
        self.connected = nn.Linear(25088, 16)
        
        #dropout
        self.dropout = nn.Dropout(0.2)
        
        
    def forward(self, x):
        #stage 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        #stage 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        #stage 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.connected(x)
        return x
        
        

#model = CNN()
weights = EfficientNet_B3_Weights.DEFAULT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = efficientnet_b3(weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features,10)
model = model.to(device)