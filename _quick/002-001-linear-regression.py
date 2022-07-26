import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Input (temp, rainfall, humidity)
inputs = torch.tensor([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype=torch.float32)

# Targets (apples, oranges)
targets = torch.tensor([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype=torch.float32)


# Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True) # this should create 3 batches each having 5 data


#Define model
model = nn.Linear(3, 2)  # inputs, targets
# Define loss Function
loss_fn = F.mse_loss
# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Check initial params for debugging
print("Weights:",model.weight)
print("Biases:",model.bias)


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl,log_freq):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
        
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % log_freq == 0:
            # print(xb,yb)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

fit(400, model, loss_fn, opt, train_dl,20)

# Generate predictions and compare with targets
preds = model(inputs)
print("Preditions:",preds)
print("Targets:",preds)

