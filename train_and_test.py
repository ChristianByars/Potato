import CNN
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch_size = 16
    #set model to training mode
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        prediction = model(X)
        loss = loss_fn(prediction, y)
        
        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>8f}    [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:.8f}\n")
    
#----------------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(CNN.model.parameters(), lr = 1e-3)

epochs = 50
for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------")
    train(CNN.train_loader, CNN.model, loss_fn, optimizer)
    test(CNN.test_loader, CNN.model, loss_fn)
print("Done")

torch.save(CNN.model.state_dict(), "cnn_model.pth")