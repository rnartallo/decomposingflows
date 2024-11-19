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

healthy1 =[]
for i in range(1,513):
    healthy1.append(np.array(pd.read_csv('~/Desktop/TopData2/RBC/rbc_1/traj_no_10'+str(format_to_three_digits(i))+'.txt',sep='\s+',header=None)))
healthy2 =[]
for i in range(1,513):
    healthy2.append(np.array(pd.read_csv('~/Desktop/TopData2/RBC/rbc_2/rbc_2_traces_10'+str(format_to_three_digits(i))+'.dat',sep='\s+',header=None)))
healthy3 =[]
for i in range(1,513):
    healthy3.append(np.array(pd.read_csv('~/Desktop/TopData2/RBC/rbc_3/rbc_3_traces_10'+str(format_to_three_digits(i))+'.dat',sep='\s+',header=None)))
healthy4 =[]
for i in range(1,513):
    healthy4.append(np.array(pd.read_csv('~/Desktop/TopData2/RBC/rbc_4/rbc_4_traces_10'+str(format_to_three_digits(i))+'.dat',sep='\s+',header=None)))
healthy5 =[]
for i in range(1,513):
    healthy5.append(np.array(pd.read_csv('~/Desktop/TopData2/RBC/rbc_5/rbc_5_traces_10'+str(format_to_three_digits(i))+'.dat',sep='\s+',header=None)))
fixed =[]
for i in range(1,513):
    fixed.append(np.array(pd.read_csv('~/Desktop/TopData2/RBC/Fixed/traj_no_10'+str(format_to_three_digits(i))+'.txt',sep='\s+',header=None)))


lags =0
for i in range(0,512):
    lags+=autocorrelation_crossing2(np.reshape(healthy1[i],(10000)))
    lags+=autocorrelation_crossing2(np.reshape(healthy2[i],(10000)))
    lags+=autocorrelation_crossing2(np.reshape(healthy3[i],(10000)))
    lags+=autocorrelation_crossing2(np.reshape(healthy4[i],(10000)))
    lags+=autocorrelation_crossing2(np.reshape(healthy5[i],(10000)))
    lags+=autocorrelation_crossing2(np.reshape(fixed[i],(10000)))
lag = lags/(512*6)


lag = int(np.round(lag))
healthy1_phase=[]
healthy2_phase =[]
healthy3_phase =[]
healthy4_phase=[]
healthy5_phase =[]
fixed_phase =[]
for i in range(0,512):
    healthy1_phase.append(time_delay_embedding(np.reshape(healthy1[i],(10000)),lag))
    healthy2_phase.append(time_delay_embedding(np.reshape(healthy2[i],(10000)),lag))
    healthy3_phase.append(time_delay_embedding(np.reshape(healthy3[i],(10000)),lag))
    healthy4_phase.append(time_delay_embedding(np.reshape(healthy4[i],(10000)),lag))
    healthy5_phase.append(time_delay_embedding(np.reshape(healthy5[i],(10000)),lag))
    fixed_phase.append(time_delay_embedding(np.reshape(fixed[i],(10000)),lag))

sets =[healthy1,healthy2,healthy3,healthy4,healthy5,fixed] 
phasesets =[healthy1_phase,healthy2_phase,healthy3_phase,healthy4_phase,healthy5_phase,fixed_phase] 
names = ['healthy1','healthy2','healthy3','healthy4','healthy5','fixed']

def shuffle_phase(phase):
    shuffled = []
    T = phase[0].shape[1]
    for n in range(0,len(phase)):
        shuffled_indices = np.random.permutation(T)
        shuff = phase[n][:, shuffled_indices]
        shuffled.append(shuff)
    return shuffled

bootstraps = 100

for counter in range(0,6):
    print(str(counter) + ' counter')
    condition_phase = phasesets[counter]
    condition = sets[counter]
    Curl =[]
    Shuffled_Curl =[]
    # All trajectories
    #Curl.append(pipeline(condition_phase))

    # Bootstrapped trajectories
    for s in range(0,bootstraps):
        print(s)
        strap = random.choices(range(0,512),k=512)
        bootstrap_phase=[]
        for i in range(0,512):
            bootstrap_phase.append(time_delay_embedding(np.reshape(condition[strap[i]],(10000)),lag))
        shuffled_phase = shuffle_phase(bootstrap_phase)
        Curl.append(pipeline(bootstrap_phase))
        Shuffled_Curl.append(pipeline(shuffled_phase))
    np.savetxt(names[counter]+"_ShuffOrig.csv", np.array(Curl), delimiter=",")
    np.savetxt(names[counter]+"_ShuffShuff.csv", np.array(Shuffled_Curl), delimiter=",")
