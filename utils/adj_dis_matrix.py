import csv
import numpy as np


def get_adj_dis_matrix(distance_df_filename, num_of_vertices, direction=False, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    
    # if node id in distance_df_file doesn't start from zero, 
    # it needs to be remap via id_filename which contains the corresponding id with sorted index.
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

        with open(distance_df_filename, 'r') as f:
            f.readline()  # 略过表头那一行
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = distance
                if not direction:
                    A[id_dict[j], id_dict[i]] = 1
                    distaneA[id_dict[j], id_dict[i]] = distance
        
        return A, distaneA # adj matrix, distance matrix
    
    else:  # distance_df_file: node id starts from zero
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i, j] = 1
                distaneA[i, j] = distance
                if not direction:
                    A[j, i] = 1
                    distaneA[j, i] = distance
        
        return A, distaneA



def get_Laplacian(A):
    '''
    compute the graph Laplacian, which can be represented as L = D − A

    Parameters
    ----------
    A: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Laplacian matrix: np.ndarray, shape (N, N)

    '''

    assert (A-A.transpose()).sum() == 0  # 首先确保A是一个对称矩阵

    D = np.diag(np.sum(A, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - A  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    return L


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))  # D是度矩阵，只有对角线上有元素

    L = D - W  # L是实对称矩阵A，有n个不同特征值对应的特征向量是正交的。

    lambda_max = eigs(L, k=1, which='LR')[0].real  # 求解拉普拉斯矩阵的最大奇异值

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def sym_norm_Adj(W):
    '''
    compute Symmetric normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N) # add self-connection
    D = np.diag(np.sum(W, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix


def norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    normalized Adj matrix: (D^hat)^{-1} A^hat; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    N = W.shape[0]
    W = W + np.identity(N)  # add self-connection
    D = np.diag(1.0/np.sum(W, axis=1))
    norm_Adj_matrix = np.dot(D, W)

    return norm_Adj_matrix


def trans_norm_Adj(W):
    '''
    compute  normalized Adj matrix

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    Symmetric normalized Laplacian: (D^hat)^1/2 A^hat (D^hat)^1/2; np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]

    W = W.transpose()
    N = W.shape[0]
    W = W + np.identity(N)  # 为邻居矩阵加上自连接
    D = np.diag(1.0/np.sum(W, axis=1))
    trans_norm_Adj = np.dot(D, W)

    return trans_norm_Adj

if __name__ == "__main__":
    adj_mx, dis_mx = get_adj_dis_matrix("../dataset/PEMS03/PEMS03.csv", 358, False, "../dataset/PEMS03/PEMS03.txt")
    print(dis_mx)
    norm_dis_mx = norm_Adj(dis_mx)
    print(norm_dis_mx)
    print(adj_mx)
