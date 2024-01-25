import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import scipy.sparse as sp

from CoraData import CoraData
from Model import GCNnet

def normalization(adjacency):
        """ L = D^(-0.5) * (A + I) * D^(-0.5)"""
        adjacency += sp.eye(adjacency.shape[0])
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()

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
x = dataset.x/dataset.x.sum(1,keepdims=True)
tensor_x = torch.from_numpy(x).to(device)
tensor_y = torch.from_numpy(dataset.y).to(device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(device)
normalize_adjacency = normalization(dataset.adjacency)
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
        train_y = torch.tensor(train_y,dtype=torch.float32)
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
        accuracy = torch.eq(predict_y, tensor_y[mask].max(1)[1]).float().mean()
    return accuracy

loss_history, val_acc_history = train()

fig = plt.figure()
# 坐标系ax1画曲线1
ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
ax1.plot(range(len(loss_history)), loss_history,
            c=np.array([255, 71, 90]) / 255.)  # c为颜色
plt.ylabel('Loss')

# 坐标系ax2画曲线2
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
ax2.plot(range(len(val_acc_history)), val_acc_history,
            c=np.array([79, 179, 255]) / 255.)
ax2.yaxis.tick_right()  # 开启右边的y坐标

ax2.yaxis.set_label_position("right")
plt.ylabel('ValAcc')

plt.xlabel('Epoch')
plt.title('Training Loss & Validation Accuracy')
plt.savefig("history.jpg")

