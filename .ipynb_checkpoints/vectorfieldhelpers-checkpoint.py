import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.linalg import cholesky


def furthest_point_sampling(point_cloud, num_samples):
    N, D = point_cloud.shape
    samples = np.zeros(num_samples, dtype=int)
    distances = np.full(N, np.inf)
    
    samples[0] = np.random.randint(0, N)
    
    for i in range(1, num_samples):
        dist_to_new_point = np.linalg.norm(point_cloud - point_cloud[samples[i-1]], axis=1)
        distances = np.minimum(distances, dist_to_new_point)
        
        samples[i] = np.argmax(distances)
    
    sampled_points = point_cloud[samples]
    return samples, sampled_points

def tesselation_mid_points(triangles,points):
    return(np.mean(points[np.array(triangles)],1))

def maximum_likelihood_drift_estimation(tesselation,Y,tau):
    no_triangles = tesselation.simplices.shape[0]
    F = np.zeros((2,no_triangles))
    bin_count = np.ones((no_triangles)) #Pseudo count in each bin
    for traj in Y:
        T = traj.shape[1]
        for t in range(0,T-1):
            bin = int(tesselation.find_simplex(traj[:,t]))
            F[:,bin] += (traj[:,t+1]-traj[:,t])/tau
            bin_count[bin] +=1
    return(np.divide(F,bin_count))

def maximum_likelihood_p_ss(tesselation,Y,tau):
    no_triangles = tesselation.simplices.shape[0]
    bin_count = np.ones((no_triangles)) #Pseudo count in each bin
    for traj in Y:
        T = traj.shape[1]
        for t in range(0,T-1):
            bin = int(tesselation.find_simplex(traj[:,t]))
            bin_count[bin] +=1
    return(bin_count/T)

def maximum_likelihood_diffusion_estimation(tesselation,Y,tau):
    no_triangles = tesselation.simplices.shape[0]
    D = np.zeros((2,2,no_triangles))
    bin_count = np.ones((no_triangles)) #Pseudo count in each bin
    for traj in Y:
        T = traj.shape[1]
        for t in range(0,T-1):
            bin = int(tesselation.find_simplex(traj[:,t]))
            D[:,:,bin] += np.outer((traj[:,t+1]-traj[:,t]),(traj[:,t+1]-traj[:,t]))/(2*tau)
            bin_count[bin] +=1
    return(np.divide(D,bin_count))


def maximum_likelihood_drift_estimation_3d(tesselation,Y,tau):
    no_tetra = tesselation.simplices.shape[0]
    F = np.zeros((3,no_tetra))
    bin_count = np.ones((no_tetra)) #Pseudo count in each bin
    for traj in Y:
        T = traj.shape[1]
        for t in range(0,T-1):
            bin = int(tesselation.find_simplex(traj[:,t]))
            F[:,bin] += (traj[:,t+1]-traj[:,t])/tau
            bin_count[bin] +=1
    return(np.divide(F,bin_count))


def maximum_likelihood_diffusion_estimation_3d(tesselation,Y,tau):
    no_tetra = tesselation.simplices.shape[0]
    D = np.zeros((3,3,no_tetra))
    bin_count = np.ones((no_tetra)) #Pseudo count in each bin
    for traj in Y:
        T = traj.shape[1]
        for t in range(0,T-1):
            bin = int(tesselation.find_simplex(traj[:,t]))
            D[:,:,bin] += np.outer((traj[:,t+1]-traj[:,t]),(traj[:,t+1]-traj[:,t]))/(2*tau)
            bin_count[bin] +=1
    return(np.divide(D,bin_count))

def drift_matrix_estimation(D_x):
    return(np.mean(D_x,2)/2)

def normalise_vector_field(F):
    no_points = F.shape[1]
    nF = np.zeros(F.shape)
    for n in range(0,no_points):
        if np.linalg.norm(F[:,n])!=0:
            nF[:,n] = F[:,n]/np.linalg.norm(F[:,n])
        else: 
            nF[:,n] = F[:,n]
    return(nF)

