import CNN
import train_and_test
import torch
import torchvision
import torchsummary
import torch.nn as nn
import matplotlib.pyplot as plt



#load model
# model = CNN.model
# model.load_state_dict(torch.load(r"C:\Users\chris\OneDrive\Desktop\SDSU\Pavement Distress\Potato\cnn_model.pth"))
# model.eval()

# print(model)

#load model
model = CNN.model(num_classes = 7)
model.load_state_dict(torch.load(r"C:\Users\chris\OneDrive\Desktop\SDSU\Pavement Distress\Potato\cnn_model.pth"))
model.train()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 50
for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------")
    
    #new dataset and new loader
    CNN.train(train_loader, model, loss_fn, optimizer)
    CNN.test(test_loader, model, loss_fn)


torch.save(model.state_dict(), "cnn_model_updated.pth")