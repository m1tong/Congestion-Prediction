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


data_path = 'YOUR_FILE_PATH'

# initialize instances matrix and adjacent matrix
def instances_ini(i):
    with gzip.open(data_path + 'xbar/' + str(i) + '/xbar.json.gz','rb') as f:
        design = json.loads(f.read().decode('utf-8'))
        
    instances = pd.DataFrame(design['instances'])
    nets = pd.DataFrame(design['nets'])

    conn=np.load(data_path + 'xbar/' + str(i) + '/xbar_connectivity.npz')
    A = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape=conn['shape'])
    A = A.__mul__(A.T)

    def buildBST(array,start=0,finish=-1):
        if finish<0:
            finish = len(array)
        mid = (start + finish) // 2
        if mid-start==1:
            ltl=start
        else:
            ltl=buildBST(array,start,mid)
        
        if finish-mid==1:
            gtl=mid
        else:
            gtl=buildBST(array,mid,finish)
            
        return((array[mid],ltl,gtl))

    congestion_data = np.load(data_path + 'xbar/' + str(i) + '/xbar_congestion.npz')
    xbst=buildBST(congestion_data['xBoundaryList'])
    ybst=buildBST(congestion_data['yBoundaryList'])
    demand = np.zeros(shape = [instances.shape[0],])
    capacity = np.zeros(shape = [instances.shape[0],])


    def getGRCIndex(x,y,xbst,ybst):
        while (type(xbst)==tuple):
            if x < xbst[0]:
                xbst=xbst[1]
            else:
                xbst=xbst[2]
                
        while (type(ybst)==tuple):
            if y < ybst[0]:
                ybst=ybst[1]
            else:
                ybst=ybst[2]
                
        return ybst, xbst

    for k in range(instances.shape[0]):
        # print(k)
        xloc = instances.iloc[k]['xloc']; yloc = instances.iloc[k]['yloc']
        i,j=getGRCIndex(xloc,yloc,xbst,ybst)
        d = 0 
        c = 0
        for l in list(congestion_data['layerList']): 
            lyr=list(congestion_data['layerList']).index(l)
            d += congestion_data['demand'][lyr][i][j]
            c += congestion_data['capacity'][lyr][i][j]
        demand[k] = d
        capacity[k] = c
            
    instances['routing_demand'] = demand
    instances['routing_capacity'] = capacity
    instances['congestion'] = demand - capacity
    return instances, A

# get cell information as dataframe for degree calculation
def cell_ini(i):
    with gzip.open(data_path + 'cells.json.gz','rb') as f:
        cells = json.loads(f.read().decode('utf-8'))

    cells = pd.DataFrame(cells)
    return cells

# normalize features
def norm_ohe(new_instances):
    onehotencoder = OneHotEncoder()
    orient = onehotencoder.fit_transform(new_instances[['orient']]).toarray()
    scaler = MinMaxScaler()
    new_instances = scaler.fit_transform(new_instances[['xloc', 'yloc', 'congestion', 'width', 'height']])
    #test for non transforming
    # new_instances = new_instances[['xloc', 'yloc', 'congestion', 'width', 'height']]
    return new_instances, orient


def clean(path):
    data = []
    instance_lst = []
    for i in range(0, 13):
        print(i)
        instances, A = instances_ini(i+1)
        instance_lst.append(instances)
        cells = cell_ini(i+1)
        new_instances = pd.merge(instances, cells[['id', 'width', 'height']], left_on='cell', right_on='id', how='left')
        new_instances = new_instances[['xloc', 'yloc', 'congestion', 'width', 'height', 'orient']]
        new_instances, orient = norm_ohe(new_instances)
        y = new_instances[:,2]
        # remove congestion from new_instances
        node_features = np.delete(new_instances, 2, 1)
        node_features = np.concatenate((node_features, orient), axis=1)
        source_nodes, target_nodes = np.nonzero(A)
        edge_index = np.vstack((source_nodes, target_nodes))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        reverse_edge_index = torch.stack((edge_index[1], edge_index[0]), dim=0)  # Flip each edge
        bidirectional_edge_index = torch.cat((edge_index, reverse_edge_index), dim=1)

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        data.append(Data(x=x, edge_index=bidirectional_edge_index, y=y))
    return data


