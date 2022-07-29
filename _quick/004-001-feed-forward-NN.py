# # Training Deep Neural Networks on a GPU with PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

dataset = MNIST(root='data/', download=True, transform=ToTensor())

frac_train = 0.80
frac_val = 0.20
val_size = int(len(dataset)*frac_val)
train_size = int(len(dataset) - val_size)
train_ds, val_ds = random_split(dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))

batch_size = 128
train_loader = DataLoader(train_ds, batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)

# Let's define the model by extending the `nn.Module` class from PyTorch.

class MnistModel(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['val_loss'], result['val_acc']))

# We also need to define an `accuracy` function which calculates the accuracy of the model's prediction on an batch of inputs. It's used in `validation_step` above.


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# We'll create a model that contains a hidden layer with 32 activations.
input_size = 28*28
hidden_size = 32  # you can change this
num_classes = 10

model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

# Using GPU
print(torch.cuda.is_available())  # Check if CUDA is available


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()
print("Using Device", device)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


for images, labels in train_loader:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break

# Finally, we define a `DeviceDataLoader` class to wrap our existing data loaders and
# move batches of data to the selected device. Interestingly, we don't need to extend
# an existing class to create a PyTorch datal oader.All we need is an `__iter__` method
# to retrieve batches of data and an `__len__` method to get the number of batches.


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

# Tensors moved to the GPU have a `device` property which includes that word `cuda`. Let's verify this by looking at a batch of data from `valid_dl`.

# ## Training the Model
# We'll define two functions: `fit` and `evaluate` to train the model using gradient descent
# and evaluate its performance on the validation set.


def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
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

# Before we train the model, we need to ensure that the data and the model's parameters
# (weights and biases) are on the same device (CPU or GPU). We can reuse the `to_device`
# function to move the model's parameters to the right device.


# Model (on GPU)
model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)
to_device(model, device)

history = [evaluate(model, val_loader)]
history += fit(5, 0.5, model, train_loader, val_loader)
history += fit(5, 0.1, model, train_loader, val_loader)
losses = [x['val_loss'] for x in history]

plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs')

accuracies = [x['val_acc'] for x in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')

# Define test dataset
test_dataset = MNIST(root='data/',
                     train=False,
                     transform=ToTensor())

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

img, label = test_dataset[0]
print('Label:', label, ', Predicted:', predict_image(img, model))
img, label = test_dataset[1839]
print('Label:', label, ', Predicted:', predict_image(img, model))
img, label = test_dataset[193]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

test_loader = DeviceDataLoader(DataLoader(
    test_dataset, batch_size=256), device)
result = evaluate(model, test_loader)
print(result)
torch.save(model.state_dict(), 'mnist-feedforward.pth')