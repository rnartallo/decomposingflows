import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Delaunay
import SimplicialComplexDecomp as SCD
import vectorfieldhelpers as VF
import importlib
import IsotropicOU as IOU


def transition_rate_estimation(tesselation,Y):
    no_triangles = tesselation.simplices.shape[0]
    T = np.zeros((no_triangles,no_triangles))
    bin_count = np.ones((no_triangles))
    for t in range(0,no_triangles):
        neighbourhood = tesselation.neighbors[t]
        for n in neighbourhood:
            if n>=0:
                T[n,t]=1 #Pseudo count each transition
    
    for traj in Y:
        Time = traj.shape[1]
        for t in range(0,Time-1):
            start_bin = int(tesselation.find_simplex(traj[:,t]))
            end_bin = int(tesselation.find_simplex(traj[:,t+1]))
            neighbourhood = tesselation.neighbors[start_bin]

            if (end_bin in neighbourhood):
                T[end_bin,start_bin]+=1
                bin_count[start_bin]+=1
            elif end_bin==start_bin:
                bin_count[start_bin]+=1
    
    T = T/bin_count[:, np.newaxis]
    T-= np.diag(np.sum(T,0))
    return(T)

def edge_flow_from_transition_matrix(T,edges):
    f = np.zeros(len(edges))
    for edge in range(0,len(edges)):
        e = edges[edge]
        if T[e[0],e[1]]>0:
            f[edge] = np.log(np.sqrt(T[e[0],e[1]]/T[e[1],e[0]]))
    return f



