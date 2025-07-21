import torch
from torch import nn
import numpy as np
import pandas as pd
from idx2numpy import convert_from_file
from torch.utils.data import DataLoader, TensorDataset

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)

        # Compute dummy input to get Linear Layer input features.
        with torch.no_grad():
            dummy_input = torch.tensor(np.random.randn(1, *input_dim)).to(torch.float32)
            x = self.pool1(self.conv1(dummy_input))
            x = self.pool2(self.conv2(x))
            flattened = x.view(1, -1)
            in_features = flattened.shape[1]

        self.l1 = nn.Linear(in_features=in_features, out_features=10)

    def forward(self, x):
        if type(x) != torch.float32:
            x = x.to(torch.float32)

        # Conv 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Conv 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten Kernels
        x = x.view(x.size(0), -1)

        # Linear Layer
        x = self.l1(x)

        # Return Logits
        return x
    

images = torch.tensor(convert_from_file("datasets/MNIST-Dataset/train-images.idx3-ubyte"))
labels = torch.tensor(convert_from_file("datasets/MNIST-Dataset/train-labels.idx1-ubyte"))
perm = torch.randperm(images.size(0))
images = images[perm]
labels = labels[perm]
train_images = images[:40000, :, :].unsqueeze(1)
train_labels = labels[:40000]
eval_images = images[40000:, :, :].unsqueeze(1)
eval_labels = labels[40000:]


model = ConvolutionNeuralNetwork(input_dim=(1, 28, 28))

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.1)
loss_fn = nn.CrossEntropyLoss()


train_dataset = TensorDataset(train_images, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

eval_dataset = TensorDataset(eval_images, eval_labels)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

epochs = 5

print("Starting Training...")
for epoch in range(epochs):
    working_train_loss = 0
    train_correct = 0
    total_train = 0

    model.train()
    for image_batch, label_batch in train_dataloader:
        optimizer.zero_grad()
        logits = model(image_batch)
        loss = loss_fn(logits, label_batch)
        loss.backward()
        optimizer.step()

        total_train += len(image_batch)
        working_train_loss += float(loss)
        predicted = logits.max(1)
        train_correct += (predicted.indices == label_batch).sum().item()



    print(f"\nEpoch: {epoch}, Total-Train-Loss: {working_train_loss / total_train}, Accuracy: {train_correct / total_train}")

    working_eval_loss = 0
    eval_correct = 0
    with torch.no_grad():
        model.eval()
        for image_batch, label_batch in eval_dataloader:
            logits = model(image_batch)
            loss = loss_fn(logits, label_batch)

            predicted = logits.max(1)
            working_eval_loss += loss.item()
            eval_correct += (predicted.indices == label_batch).sum().item()



        print(f"Epoch: {epoch}, Avg-Eval-Loss: {working_eval_loss / len(eval_dataloader)}, Accuracy: {eval_correct / len(eval_dataloader)}\n")








