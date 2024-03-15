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
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, kendalltau

# You should now have instances.csv outputed by data_clean.py
# now we will do feature enginnering to the data

# feature engineering for 13 files
datas = []
for i in range(1, 14):
    # read through csv lists and output <data> for model input
    new_instances = pd.read_csv(str(i) + '_new_instances.csv')
    edge_index = np.loadtxt(str(i) +'_edge_index.csv', delimiter=',')
    # normalize xloc, yloc, congestion, width, height
    # use one hot encoding for orient
    onehotencoder = OneHotEncoder()
    orient = onehotencoder.fit_transform(new_instances[['orient']]).toarray()

    scaler = MinMaxScaler()
    node_features = scaler.fit_transform(new_instances[['xloc', 'yloc', 'width', 'GRC_density', 'node_degree']])
    node_features = np.concatenate((node_features, orient), axis=1)
    demand = new_instances[['routing_demand']].values
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(demand, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    datas.append(data)


# GCN setup
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.6):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Add self-loops to the edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)

        return x
    
#GAT setup
class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.6):
        super(GATConv, self).__init__(node_dim=0, aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformation matrices
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        # Attention coefficients
        self.attention = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index):
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, size=None)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = (x * self.attention).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Sample dropout for attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        return aggr_out

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv3(x, edge_index)
        return x
    
# set up for masks
def mask_setup(data):
    # set random seed
    torch.manual_seed(0)

    num_nodes =  node_features.shape[0]
    train_size = int(num_nodes * 0.8)  # Let's say 80% for training
    val_size = int(num_nodes * 0.1)  # 10% for validation

    # Create a random permutation of node indices
    perm = torch.randperm(num_nodes)

    # Use the first 80% of randomly permuted indices for training
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:train_size]] = True

    # Next 10% for validation
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[perm[train_size:train_size+val_size]] = True

    # Last 10% for testing
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[perm[train_size+val_size:]] = True

    # Assign masks to your data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask


def gcn_train(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GCN_model = GCN(in_channels=9, hidden_channels=9, out_channels=1, dropout=0).to(device)
    GCN_optimizer = torch.optim.Adam(GCN_model.parameters(), lr=0.005, weight_decay=5e-4)

    GCN_loss_values = []
    GCN_valid_loss_values = []

    GCN_model.train()
    for epoch in range(300):
        GCN_optimizer.zero_grad()
        GCN_out = GCN_model(data)
        GCN_loss = F.mse_loss(GCN_out[data.train_mask], data.y[data.train_mask])
        GCN_valid_loss = F.mse_loss(GCN_out[data.val_mask], data.y[data.val_mask])
        GCN_loss.backward()
        GCN_optimizer.step()
        GCN_loss_values.append(GCN_loss.item())
        GCN_valid_loss_values.append(GCN_valid_loss.item())
        print(f'Epoch {epoch+1}, Loss: {GCN_loss.item()}')
    return GCN_model

def gat_train(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GAT_model = GAT(in_channels=9, hidden_channels=9, out_channels=1, heads=9, dropout=0).to(device)
    GAT_optimizer = torch.optim.Adam(GAT_model.parameters(), lr=0.005, weight_decay=5e-4)

    GAT_loss_values = []
    GAT_valid_loss_values = []

    GAT_model.train()
    for epoch in range(300):
        GAT_optimizer.zero_grad()
        GAT_out = GAT_model(data)
        GAT_loss = F.mse_loss(GAT_out[data.train_mask], data.y[data.train_mask])
        GAT_valid_loss = F.mse_loss(GAT_out[data.val_mask], data.y[data.val_mask])
        GAT_loss.backward()
        GAT_optimizer.step()
        GAT_loss_values.append(GAT_loss.item())
        GAT_valid_loss_values.append(GAT_valid_loss.item())
        print(f'Epoch {epoch+1}, Loss: {GAT_loss.item()}')
    return GAT_model

def gcn_eval(GCN_model, data):
    # Evaluate the model
    GCN_model.eval()
    GCN_out =GCN_model(data)
    GCN_loss = F.mse_loss(GCN_out[data.test_mask], data.y[data.test_mask])
    print(f'GCN_Test Loss: {GCN_loss.item()}')
    return GCN_loss, GCN_out

def gat_eval(GAT_model, data):
    # Evaluate the model
    GAT_model.eval()
    GAT_out =GAT_model(data)
    GAT_loss = F.mse_loss(GAT_out[data.test_mask], data.y[data.test_mask])
    print(f'GAT_Test Loss: {GAT_loss.item()}')
    return GAT_loss

if __name__ == "__models__":
    # Assuming 'data' is initialized somewhere in your code
    
    # Set up masks
    mask_setup(data)
    
    # User choice: either 'gcn' or 'gat'
    model_to_run = input("Enter 'gcn' to run GCN or 'gat' to run GAT: ").lower()
    
    if model_to_run == 'gcn':
        gcn_model = gcn_train(data)
        gcn_test_loss, GCN_out = gcn_eval(gcn_model, data)
        
        # Calculate RMSE for GCN
        GCN_rmse = mean_squared_error(data.y[data.test_mask].cpu().detach().numpy(), GCN_out[data.test_mask].cpu().detach().numpy(), squared=False)
        
        # Plot the distribution of original congestion and predicted congestion for GCN
        plt.figure(figsize=(10, 5))
        sns.histplot(data.y[data.test_mask].cpu().detach().numpy(), kde=True, label='Ground Truth Congestion', stat="density", color='blue')
        sns.histplot(GCN_out[data.test_mask].cpu().detach().numpy().flatten(), kde=True, label='Predicted Congestion', stat="density", color='red')
        plt.xlabel('Congestion')
        plt.ylabel('Frequency')
        plt.title('GCN Prediction vs Ground Truth')
        plt.legend()
        plt.show()
        
    elif model_to_run == 'gat':
        gat_model = gat_train(data)
        gat_test_loss, GAT_out = gat_eval(gat_model, data)
      
                # Calculate RMSE for GCN
        GAT_rmse = mean_squared_error(data.y[data.test_mask].cpu().detach().numpy(), GAT_out[data.test_mask].cpu().detach().numpy(), squared=False)
        
        # Plot the distribution of original congestion and predicted congestion for GCN
        plt.figure(figsize=(10, 5))
        sns.histplot(data.y[data.test_mask].cpu().detach().numpy(), kde=True, label='Ground Truth Congestion', stat="density", color='blue')
        sns.histplot(GAT_out[data.test_mask].cpu().detach().numpy().flatten(), kde=True, label='Predicted Congestion', stat="density", color='red')
        plt.xlabel('Congestion')
        plt.ylabel('Frequency')
        plt.title('GAT Prediction vs Ground Truth')
        plt.legend()
        plt.show()
        # Plotting or further evaluation for GAT can be added here
        
    else:
        print("Invalid input! Please enter 'gcn' or 'gat'.")



