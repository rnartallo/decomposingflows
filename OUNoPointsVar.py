import numpy as np
import scipy as sp
from OldExamples import Ex2 as EX2
from scipy.stats import multivariate_normal
import SimplicialComplexDecomp as SCD
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
import vectorfieldhelpers as VF
import OURotational as OU_R
import IsotropicOU as IOU
import MarkovChainHelpers as MC

def pipeline(sampled_points):
    
    no_trajectories=500
    theta=4
    samples=10000
    tau=0.01
    eps=0.5
    Y =[]
    for i in range(0,no_trajectories):
        x0 = np.random.normal(0,1,2)
        X, X_rev, X_irrev, X_check = IOU.isotropic_OU2(theta,eps,x0,tau,samples)
        Y.append(X)
    U =np.concatenate(Y, axis=1)
    subUidx,subU  = VF.furthest_point_sampling(U.T,sampled_points)
    subU = subU.T

    tesselation = Delaunay(subU.T)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex(subU.T,tesselation.simplices)
    mid_points = VF.tesselation_mid_points(triangles,tesselation.points)
    T = MC.transition_rate_estimation(tesselation,Y)

    tesselation2 = Delaunay(mid_points)
    vertices, edges, triangles = SCD.tesselation_to_simplicial_complex(mid_points,tesselation2.simplices)
    ef = MC.edge_flow_from_transition_matrix(T,edges)
    X_curl, X_grad = SCD.helmholtz_decomposition_sparse_fast(ef,vertices,edges)
    C,G = SCD.proportion_of_flow_no_h(X_curl.reshape((len(edges))),X_grad.reshape((len(edges))),ef)

    print(C)
    print(G)
    return([C,G])

points_range = np.linspace(50,500,10)
Curl =[]

for e in range(0,10):
    print(e)
    no_points = points_range[e]
    [C,G] = pipeline(int(no_points))
    Curl.append(C)


np.savetxt("CurlIsoOU2NoPoints.csv", np.array(Curl), delimiter=",")