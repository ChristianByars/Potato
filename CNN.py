import os
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import collections.abc
collections.Iterable = collections.abc.Iterable


#number of classifications and input shape
num_classes = 7
input_size = 3*224*224

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ]
)

train_path = r"C:\Users\chris\OneDrive\Desktop\SDSU\Pavement Distress\PotatoCNN\Dataset/train"
test_path = r"C:\Users\chris\OneDrive\Desktop\SDSU\Pavement Distress\PotatoCNN\Dataset/test"

train_dataset = datasets.ImageFolder(root = train_path, transform=transform)
test_dataset = datasets.ImageFolder(root = test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size= 16, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size= 16, shuffle = True)

#image shape: [3,224,224]
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.relu_stack(x)
        return logits

model = CNN()