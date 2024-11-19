import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vectorfieldhelpers as VF
import SimplicialComplexDecomp as SCD
from scipy.spatial import Delaunay
import importlib
import MarkovChainHelpers as MC
import random 

def time_delay_embedding(x,lag):
    T = len(x)
    X = np.zeros((2,T-lag))
    for t in range(0,T-lag):
        X[:,t] = [x[t],x[t+lag]]
    return X

def format_to_three_digits(number):
    return f"{number:03d}"

def autocorrelation(time_series, lag):
    mean = np.mean(time_series)
    n = len(time_series)
    variance = np.var(time_series)
    
    cov = np.sum((time_series[:n-lag] - mean) * (time_series[lag:] - mean)) / n
    
    return cov / variance

def autocorrelation_crossing2(series):
    lag = 1
    a = autocorrelation(series,lag)
    while a>0:
        lag+=1
        a = autocorrelation(series,lag)
    return lag


def pipeline(phase):
    U =np.concatenate(phase, axis=1)
    subUidx,subU  = VF.furthest_point_sampling(U.T,250)
    subU = subU.T
    tesselation = Delaunay(subU.T)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex(subU.T,tesselation.simplices)
    D_x = VF.maximum_likelihood_diffusion_estimation(tesselation,phase,tau=0.01)
    D_const = VF.drift_matrix_estimation(D_x)
    Dsqrt = sp.linalg.sqrtm(D_const)
    temp = phase
    phase =[]
    # Coordinate change for isotropic noise
    for obs in temp:
        phase.append(Dsqrt@obs)
    U =np.concatenate(phase, axis=1)
    subUidx,subU  = VF.furthest_point_sampling(U.T,250)
    subU = subU.T
    tesselation = Delaunay(subU.T)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex(subU.T,tesselation.simplices)
    mid_points = VF.tesselation_mid_points(triangles,tesselation.points)
    T = MC.transition_rate_estimation(tesselation,phase)
    tesselation2 = Delaunay(mid_points)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex(mid_points,tesselation2.simplices)
    ef = MC.edge_flow_from_transition_matrix(T,edges)
    Y = mid_points.T
    X_curl, X_grad = SCD.helmholtz_decomposition_sparse_fast(ef,vertices,edges)
    C,G = SCD.proportion_of_flow_no_h(X_curl.reshape((len(edges))),X_grad.reshape((len(edges))),ef)
    return C

#ppt = random.sample(range(0,10506),4046)
normal = np.array(pd.read_csv('~/Desktop/TopData2/Heart/ptbdb_normal.csv',header=None))
abnormal = np.array(pd.read_csv('~/Desktop/TopData2/Heart/ptbdb_abnormal.csv',header=None))
normal_ensemble =[]
abnormal_ensemble =[]
for i in range(0,len(ppt)):
    normal_ensemble.append(normal[i,:])
    abnormal_ensemble.append(abnormal[i,:])
lags =0
for i in range(0,4046):
    lags+=autocorrelation_crossing2(normal_ensemble[i])
    lags+=autocorrelation_crossing2(abnormal_ensemble[i])
lag = lags/(2*4046)
lag = int(np.round(lag))
normal_phase=[]
abnormal_phase =[]
for i in range(0,4046):
    normal_phase.append(time_delay_embedding(normal_ensemble[i],lag))
    abnormal_phase.append(time_delay_embedding(abnormal_ensemble[i],lag))

sets =[normal_ensemble,abnormal_ensemble] 
names = ['normal','abnormal']
phasesets =[normal_phase,abnormal_phase] 

bootstraps = 100

for counter in range(0,2):
    print(str(counter) + ' counter')
    condition_phase = phasesets[counter]
    condition = sets[counter]
    Curl =[]
    # All trajectories
    Curl.append(pipeline(condition_phase))

    # Bootstrapped trajectories
    for s in range(0,bootstraps):
        print(s)
        strap = random.choices(range(0,4046),k=4046)
        bootstrap_phase=[]
        for i in range(0,4046):
            bootstrap_phase.append(time_delay_embedding(condition[strap[i]],lag))
        Curl.append(pipeline(bootstrap_phase))
    np.savetxt(names[counter]+".csv", np.array(Curl), delimiter=",")
                            
np.savetxt('ppt.csv',np.array(ppt),delimiter=",")
