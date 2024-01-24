import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from CoraData import CoraData
from Model import GCNnet


learning_rate = 0.1
weight_decay = 5e-4
epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GCNnet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), 
                       lr=learning_rate,
                       weight_decay=weight_decay)
dataset = CoraData().data
x = dataset.x/dataset.x.sum(1, keepdim=True)
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalize_adjacency = torch.from_numpy(dataset.adjacency).to(device)
indices = torch.from_numpy(np.asarray(
    [normalize_adjacency.row,normalize_adjacency.col]
    )).long()
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices,values,
                                            (2708,2708)).to(device)

def train():
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)
        train_mask_logits = logits[tensor_train_mask]
        loss = criterion(train_mask_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = test(tensor_train_mask)
        val_acc = test(tensor_val_mask)
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4f}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(),val_acc.item()
        ))

    return loss_history, val_acc_history

def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    return accuracy

