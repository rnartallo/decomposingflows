import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import vectorfieldhelpers as VF
import SimplicialComplexDecomp as SCD
from scipy.spatial import Delaunay
import MarkovChainHelpers as MC

def tamed_euler_rossler(theta,eps,h,samples,x0):
    X = np.zeros((3,samples))
    a = 0.2
    b = 0.2 
    c = 5.7
    X[:,0] = x0
    for t in range(0,samples-1):
        x,y,z = X[:,t]
        norm = np.linalg.norm(np.array([-theta*y-z,theta*x+a*y,b+z*(x-c)]))
        x = x + (h*(-theta*y-z))/(1+h*norm)+eps*np.random.normal(loc=0,scale=np.sqrt(h))
        y = y + (h*(theta*x+a*y))/(1+h*norm)+eps*np.random.normal(loc=0,scale=np.sqrt(h))
        z = z + (h*(b+z*(x-c)))/(1+h*norm)+eps*np.random.normal(loc=0,scale=np.sqrt(h))
        X[:,t+1] = np.array([x,y,z])
    return X

def pipeline(theta):

    no_trajectories = 500
    samples=20000
    h=0.01
    eps = 0.075
    Y =[]
    for i in range(0,no_trajectories):
        x0 = np.random.normal(0,1,3)
        Y.append(tamed_euler_rossler(theta,eps,h,samples,x0))
    U =np.concatenate(Y, axis=1)
    subUidx,subU  = VF.furthest_point_sampling(U.T,250)
    subU = subU.T

    tesselation = Delaunay(subU.T)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex_tetra(subU.T,tesselation.simplices)
    mid_points = VF.tesselation_mid_points(tesselation.simplices,tesselation.points)

    T = MC.transition_rate_estimation(tesselation,Y)

    tesselation2 = Delaunay(mid_points)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex_tetra(mid_points,tesselation2.simplices)

    ef = MC.edge_flow_from_transition_matrix(T,edges)
    X_curl, X_grad = SCD.helmholtz_decomposition_sparse_fast(ef,vertices,edges)
    C,G = SCD.proportion_of_flow_no_h(X_curl.reshape((len(edges))),X_grad.reshape((len(edges))),ef)

    print(C)
    print(G)
    return([C,G])

theta_range = np.linspace(1,4,100)
Curl =[]
Grad =[]

for e in range(0,100):
    print(e)
    theta = theta_range[e]
    [C,G] = pipeline(theta)
    Curl.append(C)
    Grad.append(G)


np.savetxt("CurlRosslerMC2.csv", np.array(Curl), delimiter=",")
np.savetxt("GradRosslerMC2.csv", np.array(Grad), delimiter=",")
