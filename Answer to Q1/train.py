# %%
from model import Model
import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
from model import Model
from torchvision.transforms import ToTensor

import torch
import torch.utils.data
from early_stop import EarlyStopping

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=1, num_features=128,num_instances=100,num_bins=21)

# %%
training_data = datasets.MNIST(
    root="/home/ubuntu/MLPackageStudy/data",
    train=True,
    download=False,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="/home/ubuntu/MLPackageStudy/data",
    train=False,
    download=False,
    transform=ToTensor()
)

# %%
training_data_filter = [(i,j) for (i,j) in training_data if j==0 or j==7]
test_data_filter = [(i,j) for (i,j) in test_data if j==0 or j==7]
labels_map = np.arange(0,10,1)
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data_filter), size=(1,)).item()
    img, label = training_data_filter[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %%
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.reshape([-1,1,28,28])
        print(X.shape)
        pred = model(X)
        # print(pred)
        y = torch.tensor(y)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.reshape([-1,1,28,28])
            pred = model(X)
            y = torch.tensor(y)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

# %%
from typing import List, Dict, Tuple
import copy
def data_generation(instance_index_label: List[Tuple]) -> List[Dict]:
    """
    bags: {key1: [ind1, ind2, ind3],
           key2: [ind1, ind2, ind3, ind4, ind5],
           ... }
    bag_lbls:
        {key1: 0,
         key2: 1,
         ... }
    """
    bag_size = np.ones(len(instance_index_label)//20,dtype = np.int)+99
    data_cp = copy.copy(instance_index_label)
    np.random.shuffle(data_cp)
    bags = {}
    bags_per_instance_labels = {}
    bags_labels = {}
    for bag_ind, size in enumerate(bag_size):
        bags[bag_ind] = []
        bags_per_instance_labels[bag_ind] = []
        p = np.random.randint(0,101)

        try:
            for _ in range(size):
                r = np.random.randint(0,len(instance_index_label))
                inst_ind, lbl = data_cp[r][0],data_cp[r][1]
                while p>0 and lbl ==7:
                    r = np.random.randint(0,len(instance_index_label))
                    inst_ind, lbl = data_cp[r][0],data_cp[r][1]
                while p==0 and lbl ==0:
                    r = np.random.randint(0,len(instance_index_label))
                    inst_ind, lbl = data_cp[r][0],data_cp[r][1]
                if p>0 and lbl==0:
                    p-=1    
                bags[bag_ind].append(inst_ind)
                # simplfy, just use a temporary variable instead of bags_per_instance_labels
                bags_per_instance_labels[bag_ind].append(lbl)
            bags_labels[bag_ind] = bag_label_from_instance_labels(bags_per_instance_labels[bag_ind])
        except:
            break
    return bags, bags_labels

def bag_label_from_instance_labels(instance_labels):
    return int(sum(((x==0) for x in instance_labels)))/100
from torch.utils.data import Dataset
class Transform_data(Dataset):
    """
    We want to 1. pad tensor 2. transform the data to the size that fits in the input size.
    
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        
    def __getitem__(self, index):
        tensor = self.data[index][0]
        if self.transform is not None:
            tensor = self.transform(tensor)
        return (tensor, self.data[index][1])

    def __len__(self):
        return len(self.data)

def pad_tensor(data:list, max_number_instance) -> list:
    """
    Since our bag has different sizes, we need to pad each tensor to have the same shape (max: 7).
    We will look through each one instance and look at the shape of the tensor, and then we will pad 7-n 
    to the existing tensor where n is the number of instances in the bag.
    The function will return a padded data set."""
    new_data = []
    for bag_index in range(len(data)):
        tensor_size = len(data[bag_index][0])
        pad_size = max_number_instance - tensor_size
        p2d = (0,0, 0, pad_size)
        padded = nn.functional.pad(data[bag_index][0], p2d, 'constant', 0)
        new_data.append((torch.reshape(padded,(-1,28,28)), data[bag_index][1]))
    return new_data

def get_data_loaders(train_data, test_data, train_batch_size=64, val_batch_size=64):
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def construct_data(instance_index_label,data):
    bag_indices, bag_labels = data_generation(instance_index_label)
    bag_features = {kk: torch.Tensor([data[i][0].numpy() for i in inds]) for kk, inds in bag_indices.items()}
    train_data = [(bag_features[i],bag_labels[i]) for i in range(len(bag_features)-1)]
    max_number_instance = 100
    padded_data = pad_tensor(train_data, max_number_instance)
    return padded_data

# %%
instance_index_label = [(i , training_data_filter[i][1]) for i in range(len(training_data_filter))]
instance_index_label_test = [(i , test_data_filter[i][1]) for i in range(len(test_data_filter))]
padded_train = construct_data(instance_index_label,training_data_filter)
padded_test = construct_data(instance_index_label_test,test_data_filter)

# %%
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
from torch.utils.data import DataLoader

train_dataloader, test_dataloader = get_data_loaders(padded_train, padded_test)


early_stopping = EarlyStopping(patience=5, verbose=True,path="/home/ubuntu/MLPackageStudy/NTU_codes/Q2.pt")
for epoch in range(10000):
    train_loop(train_dataloader,model,loss_fn,optimizer)
    t_loss = test_loop(test_dataloader,model,loss_fn)
    early_stopping(t_loss, model)
    if early_stopping.early_stop:
        break

