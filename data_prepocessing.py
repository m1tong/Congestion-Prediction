import gzip
import json
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz

data_path = 'NCSU-DigIC-GraphData-2023-07-25/'

#Stores instances dataframe into arrays
instdf = []

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




for i in range(1, 14):
    with gzip.open(f'{data_path}/xbar/{i}/xbar.json.gz', 'rb') as f:
        design = json.loads(f.read().decode('utf-8'))

    instances = pd.DataFrame(design['instances'])
    nets = pd.DataFrame(design['nets'])


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
for i, df in enumerate(newinsdf, start=1):
    filename = f"new_instances_{i}.csv"
    df.to_csv(filename, index=False)

