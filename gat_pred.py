import networkx as nx
from node2vec import Node2Vec
import gzip
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import KFold
import torch.optim as optim
from torch_geometric.data import DataLoader 


import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import StepLR


# gat define
class DeeperGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, hidden_channels_3, out_channels, heads_1=1, heads_2=1, heads_3=1, heads_4=1, dropout=0.4):
        super(DeeperGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels_1, heads=heads_1, dropout=dropout)
        # Adjusted for multi-head output
        self.conv2 = GATConv(hidden_channels_1 * heads_1, hidden_channels_2, heads=heads_2, dropout=dropout)
        self.conv3 = GATConv(hidden_channels_2 * heads_2, hidden_channels_3, heads=heads_3, dropout=dropout)
        self.conv4 = GATConv(hidden_channels_3 * heads_3, out_channels, heads=heads_4, concat=False, dropout=dropout)  # concat=False in the last layer to not multiply output dimensions
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        return x

gat1_data =  "YOUR_DATA"

# single file gat
num_nodes = 3952  
train_size = int(num_nodes * 0.8)  
val_size = int(num_nodes * 0.1) 

perm = torch.randperm(num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:train_size]] = True

val_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask[perm[train_size:train_size+val_size]] = True

# Last 10% for testing
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[perm[train_size+val_size:]] = True

gat1_data.train_mask = train_mask
gat1_data.val_mask = val_mask
gat1_data.test_mask = test_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


gat1_model = DeeperGAT(
    in_channels=8, 
    hidden_channels_1=16, 
    hidden_channels_2=16, 
    hidden_channels_3=16, 
    out_channels=1, 
    heads_1=4, 
    heads_2=4, 
    heads_3=4, 
    heads_4=1,  
    dropout=0.4
).to(device)

optimizer = torch.optim.Adam(gat1_model.parameters(), lr=0.005, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # Adjust step_size and gamma as needed


gat1_model.train()
for epoch in range(200):
    optimizer.zero_grad()
    
    out = gat1_model(gat1_data)
    loss = F.mse_loss(out[gat1_data.train_mask], gat1_data.y[gat1_data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Optional: Evaluation phase
    gat1_model.eval()
    with torch.no_grad():
        val_out = gat1_model(gat1_data)
        val_loss = F.mse_loss(val_out[gat1_data.val_mask], gat1_data.y[gat1_data.val_mask])
    
    scheduler.step() 
    
    print(f'Epoch {epoch+1}: Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')



def gat1_eval(model, data, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        data = data.to(device)
        targets = data.y.to(device)
        out = model(data)
        test_mask = data.test_mask
        loss = F.mse_loss(out[test_mask], targets[test_mask].view_as(out[test_mask]))  # Ensure shape compatibility
        total_loss = loss.item()
        predictions = out[test_mask].cpu().detach().numpy()
    return total_loss, predictions

#gat1_test_loss, gat1_predictions = gat1_eval(gat1_model, gat1_data, device)


# GAT ACROSS FILES
train_data = "YOUR_DATA"
test_data = "YOUR_DATA"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adjusted to match DeeperGAT class
gat_model = DeeperGAT(
    in_channels=8, 
    hidden_channels_1=32, 
    hidden_channels_2=32, 
    hidden_channels_3=32, 
    out_channels=1, 
    heads_1=4, 
    heads_2=4, 
    heads_3=4, 
    heads_4=1, 
    dropout=0.4
).to(device)

optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.005, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  

loss_values = []

gat_model.train()
for epoch in range(500):
    total_loss = 0
    for graph_data in train_data:
        graph_data = graph_data.to(device)
        optimizer.zero_grad()
        out = gat_model(graph_data)
        loss = F.mse_loss(out, graph_data.y.view(-1, 1))  
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()  
    avg_loss = total_loss / len(train_data)

def gat_evaluate_model(model, data_list, device):
    model.eval()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for graph_data in data_list:
            graph_data = graph_data.to(device)
            targets = graph_data.y.to(device)
            out = model(graph_data)
            loss = F.mse_loss(out, targets.view_as(out))  # Adjusting to ensure shape compatibility
            total_loss += loss.item()
            predictions.append(out.cpu().detach().numpy())
    avg_loss = total_loss / len(data_list)
    return avg_loss, predictions

# gat_test_loss, gat_predictions = gat_evaluate_model(gat_model, test_data, device)
# print(f'Test Loss: {gat_test_loss}')

