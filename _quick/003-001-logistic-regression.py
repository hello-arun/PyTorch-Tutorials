# Imports
import torch.nn as nn
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F


# MNIST dataset (images and labels)
dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())
test_ds = MNIST(root='data/',
                     train=False,
                     transform=transforms.ToTensor())

# Split train validation split
train_ds, val_ds = random_split(dataset, [50000, 10000])
len(train_ds), len(val_ds)

# DataLoader,`shuffle=True` for the training data loader to ensure that the batches generated in each epoch are different
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


input_size = 28*28
num_classes = 10

# Logistic regression model
class MnistModel(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = self.accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

    def accuracy(self,outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []  # for recording epoch-wise results

    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# model = MnistModel(input_size,num_classes)
# result0 = evaluate(model, val_loader)
# history1 = fit(5, 0.001, model, train_loader, val_loader)
# history2 = fit(5, 0.001, model, train_loader, val_loader)
# history3 = fit(5, 0.001, model, train_loader, val_loader)
# history4 = fit(5, 0.001, model, train_loader, val_loader)

# history = [result0] + history1 + history2 + history3 + history4

# accuracies = [result['val_acc'] for result in history]
# plt.plot(accuracies, '-x')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.title('Accuracy vs. No. of epochs')
# plt.show()
# test_loader = DataLoader(test_ds, batch_size=256)
# result = evaluate(model, test_loader)
# torch.save(model.state_dict(), '003-001-mnist-logistic.pth')

# Loading again the saved model
model2 = MnistModel(input_size,num_classes)
model2.load_state_dict(torch.load('003-001-mnist-logistic.pth'))
test_loader = DataLoader(test_ds, batch_size=256)
result = evaluate(model2, test_loader)
print(result)