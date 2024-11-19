import numpy as np
import scipy as sp
import SimplicialComplexDecomp as SCD
from scipy.spatial import Delaunay
import vectorfieldhelpers as VF
import MarkovChainHelpers as MC

def strang_splitting_planar_cycle_theta(theta,eps,h,samples,x0):

    X = np.zeros((2,samples))

    A = np.array([[1,-theta],[theta,1]])
    C = np.array([[eps*np.exp(0.5*h)*np.sinh(h/2),0],[0,eps*np.exp(0.5*h)*np.sinh(h/2)]])
    Eh = sp.linalg.expm(A*0.5*h)

    X[:,0] = x0
    for t in range(0,samples-1):
        xi = np.random.multivariate_normal(mean=np.zeros((2)),cov=C)
        X_temp = Eh@X[:,t] + xi
        x,y = X_temp
        r0 = np.sqrt(x**2 + y**2)
        r = r0/(r0*h+1)
        x = (r/r0)*x
        y = (r/r0)*y
        xi = np.random.multivariate_normal(mean=np.zeros((2)),cov=C)
        X[:,t+1] = Eh@np.array([x,y]) + xi
        
    return(X)

def pipeline(theta):

    no_trajectories = 500
    samples=10000
    h=0.01
    eps = 0.5
    Y =[]
    for i in range(0,no_trajectories):
        x0 = np.random.normal(0,1,2)
        Y.append(strang_splitting_planar_cycle_theta(theta,eps,h,samples,x0))
    U =np.concatenate(Y, axis=1)
    subUidx,subU  = VF.furthest_point_sampling(U.T,250)
    subU = subU.T

    tesselation = Delaunay(subU.T)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex(subU.T,tesselation.simplices)
    mid_points = VF.tesselation_mid_points(triangles,tesselation.points)

    T = MC.transition_rate_estimation(tesselation,Y)

    tesselation2 = Delaunay(mid_points)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex(mid_points,tesselation2.simplices)
    ef = MC.edge_flow_from_transition_matrix(T,edges)

    X_curl, X_grad  = SCD.helmholtz_decomposition_sparse_fast(ef,vertices,edges)
    C,G = SCD.proportion_of_flow_no_h(X_curl.reshape((len(edges))),X_grad.reshape((len(edges))),ef)

    print(C)
    print(G)
    return([C,G])

theta_range = np.linspace(0.1,10,100)
Curl =[]
Grad =[]
CurlNorm =[]
GradNorm =[]

for e in range(0,100):
    print(e)
    theta = theta_range[e]
    [C,G] = pipeline(theta)
    Curl.append(C)
    Grad.append(G)

np.savetxt("CurlPLC_MC.csv", np.array(Curl), delimiter=",")
np.savetxt("GradPLC_MC.csv", np.array(Grad), delimiter=",")
