import numpy as np
import scipy as sp
from itertools import combinations
from scipy.sparse.linalg import lsqr
from scipy.linalg import sqrtm

def find_edges(triplets):
    edges = set()
    for triplet in triplets:
        triplet_edges = combinations(sorted(triplet), 2)
        edges.update(triplet_edges)
    edges = [list(edge) for edge in edges]
    return edges

def find_triplets(tetras):
    triangles = set()
    for tetra in tetras:
        tetra_triangles = combinations(sorted(tetra), 3)
        triangles.update(tetra_triangles)
    triangles = [list(tri) for tri in triangles]
    return triangles

def extract_kcliques(f,k):
    kcliques =[]
    for s in f:
        if len(s)==k:
            kcliques.append([i for i in s])
    return(kcliques)



def curl_adjoint_matrix(triangles,edges):
    M = len(triangles)
    N = len(edges)
    C_a = np.zeros((N,M))
    for n in range(0,N):
        for m in range(0,M):
            flat_triangle = flatten_triangle(triangles[m])
            if edges[n] in flat_triangle:
                C_a[n,m] = 1
            elif [edges[n][1],edges[n][0]] in flat_triangle:
                C_a[n,m] = -1
    return C_a

def grad_matrix(edges,vertices):
    N = len(edges)
    P = len(vertices)
    B = np.zeros((N,P))
    for n in range(0,N):
        for p in range(0,P):
            if vertices[p]==edges[n][0]:
                B[n,p]=-1
            elif vertices[p]==edges[n][1]:
                B[n,p]=1
    return(B)

def grad_matrix_sparse(edges,vertices):
    N = len(edges)
    P = len(vertices)
    B =  sp.sparse.csr_matrix((N,P))
    for n in range(0,N):
        for p in range(0,P):
            if vertices[p]==edges[n][0]:
                B[n,p]=-1
            elif vertices[p]==edges[n][1]:
                B[n,p]=1
    return(B)

def div_matrix(edges,vertices):
    N = len(edges)
    P = len(vertices)
    D = np.zeros((P,N))
    for p in range(0,P):
        for n in range(0,N):
            if vertices[p] == edges[n][0]:
                D[p,n] = 1
            elif vertices[p] == edges[n][1]:
                D[p,n] = -1
    return D


def tesselation_to_simplicial_complex(centroids,triangles):
    vertices = np.linspace(0,len(centroids)-1,len(centroids),dtype=int)
    edges = find_edges(triangles)
    return(vertices,edges,triangles)

def flatten_triangle(triangle):
    flat_triangle =[[triangle[0],triangle[1]],[triangle[1],triangle[2]],[triangle[2],triangle[0]]]
    return(flat_triangle)

def curl_matrix(triangles,edges):
    M = len(triangles)
    N = len(edges)
    C = np.zeros((M,N))
    for m in range(0,M):
        flat_triangle = flatten_triangle(triangles[m])
        for n in range(0,N):
            if edges[n] in flat_triangle:
                C[m,n] = 1
            elif [edges[n][1],edges[n][0]] in flat_triangle:
                C[m,n] = -1
    return C

def graph_helmholtzian(vertices,edges,triangles):
    C = curl_matrix(triangles,edges)
    C_a = curl_adjoint_matrix(triangles,edges)
    D = div_matrix(edges,vertices)
    G = grad_matrix(edges,vertices)

    L = -G@D+C_a@C
    return(L)


def helmholtz_decomposition(f,X):
    vertices = extract_kcliques(f,1)
    edges = extract_kcliques(f,2)
    triangles = extract_kcliques(f,3)
    A = curl_adjoint_matrix(triangles,edges)
    B = grad_matrix(edges,vertices)
    v = np.linalg.lstsq(A,X)[0]
    u = np.linalg.lstsq(B,X)[0]
    X_curl = A@v
    X_grad = B@u
    X_harmonic = X - X_curl - X_grad
    return(X_curl,X_grad,X_harmonic)

def tesselation_to_simplicial_complex_tetra(centroids,tetra):
    vertices = np.linspace(0,len(centroids)-1,len(centroids),dtype=int)
    triangles = find_triplets(tetra)
    edges = find_edges(triangles)
    return(vertices,edges,triangles)

def helmholtz_decomposition_tesselation(X,vertices, edges, triangles):
    A = curl_adjoint_matrix(triangles,edges)
    B = grad_matrix(edges,vertices)
    v = sp.linalg.lstsq(A,X)[0]
    u = sp.linalg.lstsq(B,X)[0]
    X_curl = A@v
    X_grad = B@u
    X_harmonic = X - X_curl - X_grad
    return(X_curl,X_grad,X_harmonic)


def magnitude_of_flow(X_curl,X_grad,X_harm,flow):
    flow = np.array(flow)
    curl = np.linalg.norm(X_curl)/np.linalg.norm(flow)
    grad = np.linalg.norm(X_grad)/np.linalg.norm(flow)
    harm = np.linalg.norm(X_harm)/np.linalg.norm(flow)
    return(curl,grad,harm)

def proportion_of_flow_no_h(X_curl,X_grad,flow):
    flow = np.array(flow)
    c = np.linalg.norm(X_curl)
    g = np.linalg.norm(X_grad)
    curl = c/(c+g)
    grad = g/(c+g)
    return(curl,grad)

def proportion_of_flow(X_curl,X_grad,X_harm,flow):
    flow = np.array(flow)
    c = np.linalg.norm(X_curl)
    g = np.linalg.norm(X_grad)
    h = np.linalg.norm(X_harm)
    curl = c/(c+g+h)
    grad = g/(c+g+h)
    harm = h/(c+g+h)
    return(curl,grad,harm)

def norm_of_flow(X_curl,X_grad,X_harm):
    return(np.linalg.norm(X_curl),np.linalg.norm(X_grad),np.linalg.norm(X_harm))

def helmholtz_decomposition_sparse(X,vertices, edges, triangles):
    A = curl_adjoint_matrix(triangles,edges)
    B = grad_matrix(edges,vertices)
    v = lsqr(A,X)[0]
    u = lsqr(B,X)[0]
    X_curl = np.dot(A, v)
    X_grad = np.dot(B, u)
    X_harmonic = X - X_curl - X_grad
    return(X_curl,X_grad,X_harmonic)

def helmholtz_decomposition_sparse_fast(X,vertices, edges):
    B = sp.sparse.csr_matrix(grad_matrix(edges,vertices))
    u = sp.sparse.linalg.lsqr(B,X)[0]
    X_grad = B@u
    X_curl = X - X_grad
    return(X_curl,X_grad)

def weighted_helmholtz_decomposition_sparse(X,Y,vertices, edges):
    B = sp.sparse.csr_matrix(grad_matrix(edges,vertices))
    W_sqrt = sp.sparse.csr_matrix(np.diag(np.sqrt(Y)))
    W = sp.sparse.csr_matrix(np.diag(Y))
    u = sp.sparse.linalg.lsqr(W_sqrt@B,W_sqrt@X)[0]
    X_grad = B@u
    X_curl = X - W@X_grad
    return(X_curl,X_grad,W@X_grad,u)

def weighted_helmholtz_decomposition_qian_sparse(X,Y,vertices, edges):
    B = sp.sparse.csr_matrix(grad_matrix(edges,vertices))
    W_sqrt = sp.sparse.csr_matrix(np.diag(np.sqrt(Y)))
    W = sp.sparse.csr_matrix(np.diag(Y))
    u = sp.sparse.linalg.lsqr(W_sqrt@B,W_sqrt@X)[0]
    X_grad = B@u
    X_curl = X-X_grad
    return(X_curl,X_grad,u)