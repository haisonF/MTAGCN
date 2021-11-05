import numpy as np
import tensorflow as tf
import scipy.sparse as sp

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = tf.random.uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    ) 
    return tf.Variable(initial, name=name)



def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask) 
    return pre_out*(1./keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return sparse_to_tuple(adj_nomalized)


def constructNet(miR_tar_matrix): 
    miR_matrix = np.matrix(
        np.zeros((miR_tar_matrix.shape[0], miR_tar_matrix.shape[0]), dtype=np.int8))
    tar_matrix = np.matrix(
        np.zeros((miR_tar_matrix.shape[1], miR_tar_matrix.shape[1]), dtype=np.int8))
    mat1 = np.hstack((miR_matrix, miR_tar_matrix))
    mat2 = np.hstack((miR_tar_matrix.T,tar_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def constructHNet(miR_tar_matrix, miR_matrix, tar_matrix):  
    mat1 = np.hstack((miR_matrix, miR_tar_matrix))   
    mat2 = np.hstack((miR_tar_matrix.T, tar_matrix)) 
    return np.vstack((mat1,mat2))



def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return