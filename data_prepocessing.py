import gzip
import json
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz

data_path = 'NCSU-DigIC-GraphData-2023-07-25/'

#Stores adjacency matrices and instances dataframe into arrays
instdf = []
adj = []

for i in range(1, 14):
    with gzip.open(f'{data_path}/xbar/{i}/xbar.json.gz', 'rb') as f:
        design = json.loads(f.read().decode('utf-8'))

    instances = pd.DataFrame(design['instances'])
    nets = pd.DataFrame(design['nets'])

    conn=np.load(f"{data_path}xbar/{i}/xbar_connectivity.npz")
    A = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape=conn['shape'])
    A = A.__mul__(A.T)
    adj.append(A)


    congestion_data = np.load(f"{data_path}xbar/{i}/xbar_congestion.npz")
    xbst=buildBST(congestion_data['xBoundaryList'])
    ybst=buildBST(congestion_data['yBoundaryList'])
    demand = np.zeros(shape = [instances.shape[0],])
    capacity = np.zeros(shape = [instances.shape[0],])


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
    instdf.append(instances)
#Opens dataframe with information about each cell type's width and hiehght
with gzip.open(data_path + 'cells.json.gz','rb') as f:
    cells = json.loads(f.read().decode('utf-8'))
cells = pd.DataFrame(cells)

#Stores merged instances dataframes
newinsdf = []

#Makes merged dataframe with each instances dataframe in the instdf array
for i in instdf:
    newinsdf.append(pd.merge(i, cells[['id', 'width', 'height']], left_on='cell', right_on='id', how='left'))
    
#Saves each merged dataframe into a csv
for i, df in enumerate(instdf, start=1):
    filename = f"new_instances_{i}.csv"
    df.to_csv(filename, index=False)

#Saves each adjacency matrix into npz
for i, sparse_matrix in enumerate(adj):
    filename = f"sparse_matrix_{i+1}.npz"
    sparse.save_npz(filename, sparse_matrix)