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


# data processing function
def data_clean(data_path):
    file_paths = []
    for i in range(13):
        file_paths.append('xbar/' + str(i+1))

    for num, file_path in enumerate(file_paths):
        print(file_path)
        with gzip.open(data_path + file_path + '/xbar.json.gz','rb') as f:
            design = json.loads(f.read().decode('utf-8'))
            
        instances = pd.DataFrame(design['instances'])
        nets = pd.DataFrame(design['nets'])

        conn=np.load(data_path + file_path + '/xbar_connectivity.npz')
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

        congestion_data = np.load(data_path + file_path + '/xbar_congestion.npz')
        xbst=buildBST(congestion_data['xBoundaryList'])
        ybst=buildBST(congestion_data['yBoundaryList'])
        demand = np.zeros(shape = [instances.shape[0],])
        capacity = np.zeros(shape = [instances.shape[0],])
        GRC_x = np.zeros(shape = [instances.shape[0],])
        GRC_y = np.zeros(shape = [instances.shape[0],])

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
            GRC_x[k] = i
            GRC_y[k] = j

        instances['routing_demand'] = demand
        instances['routing_capacity'] = capacity
        instances['GRC_x'] = GRC_x
        instances['GRC_y'] = GRC_y

        GRC_density = np.zeros([congestion_data['yBoundaryList'].shape[0],congestion_data['xBoundaryList'].shape[0]])
        for i in range(58):
            for j in range(57):
                GRC_density[i,j] = instances[(instances['GRC_x']==i) & (instances['GRC_y']==j)].shape[0]

        instances['GRC_density'] = instances.apply(lambda x: GRC_density[int(x['GRC_x']),int(x['GRC_y'])], axis=1)

        in_degree = A.sum(axis=0)
        in_degree = np.array(in_degree).flatten()
        instances['node_degree'] = in_degree

        with gzip.open(data_path + 'cells.json.gz','rb') as f:
            cells = json.loads(f.read().decode('utf-8'))

        cells = pd.DataFrame(cells)

        new_instances = pd.merge(instances, cells[['id', 'width', 'height']], left_on='cell', right_on='id', how='left')

        new_instances = new_instances[['routing_demand','xloc', 'yloc', 'orient', 'width','GRC_density','node_degree']]

        source_nodes, target_nodes = np.nonzero(A)
        edge_index = np.vstack((source_nodes, target_nodes))

        # save new_instances, edge_index to csv
        new_instances.to_csv(str(num+1) + '_new_instances.csv', index=False)
        np.savetxt(str(num+1) + '_edge_index.csv', edge_index, delimiter=',')




# include the file path of your IC layout data
data_path = 'FILE_PATH'
data_clean(data_path)

# Using the below code to read the instances you want to create
new_instances = pd.read_csv("INSTANCE_FILE")
edge_index = np.loadtxt('EDGE_INDEX_FILE', delimiter=',')


