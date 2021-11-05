import numpy as np
import scipy.sparse as sp
import tensorflow as tf 
import gc
import random
from clac_metric import cv_model_evaluate
from tensorflow.python.framework import ops
from utils import *
from model import GCNModel
from opt import Optimizer 
import pandas as pd


             
def PredictScore(train_miR_tar_matrix, miR_matrix, tar_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    ops.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_miR_tar_matrix, miR_matrix, tar_matrix)  
    adj = sp.csr_matrix(adj) 
    association_nam = train_miR_tar_matrix.sum()  
    X = constructNet(train_miR_tar_matrix) 
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_miR_tar_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))
    adj_norm = preprocess_graph(adj) 
    adj_nonzero = adj_norm[1].shape[0]

    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'adjdp': tf.compat.v1.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_miR_tar_matrix.shape[0], name='TMTGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_miR_tar_matrix.shape[0], num_v=train_miR_tar_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()                                                                                    
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res



def cross_validation_experiment(miR_tar_matrix, miR_matrix, tar_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(miR_tar_matrix == 1)) 
    random_index = index_matrix.T.tolist()
    label=np.array(random_index)
    
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label[i][j]=label[i][j]+1
  
    reorder = np.arange(label.shape[0])
    random.seed(seed)
    random.shuffle(random_index)  
    random.shuffle(reorder)
    k_folds = 5  # 5
    random_index=div_list(random_index,k_folds)
    reorder=div_list(reorder,k_folds)
    metric = np.zeros((1, 7)) 
    print("seed=%d, evaluating miRNA-target...." % (seed))
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(miR_tar_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0  
        miR_len = miR_tar_matrix.shape[0]
        test_arr=reorder[k]
        tar_len = miR_tar_matrix.shape[1]
        miR_tar_res = PredictScore(
            train_matrix, miR_matrix, tar_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = miR_tar_res.reshape(miR_len, tar_len) 
        metric_tmp = cv_model_evaluate(
            miR_tar_matrix, predict_y_proba, train_matrix,test_arr,label)
        print(metric_tmp)
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print(metric / k_folds) 
    metric = np.array(metric / k_folds)
    return metric



if __name__ == "__main__":
    miR_sim = np.loadtxt('../data/mirna_sim.csv', delimiter=',')    
    tar_sim = np.loadtxt('../data/target_sim.csv', delimiter=',')            
    miR_tar_matrix = np.loadtxt('../data/MTadj.csv', delimiter=',')   
    epoch=200 
    emb_dim =64                                                                                                              
    lr = 0.01    
    adjdp = 0.3
    dp=0.6
    simw=0.06
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)   
    circle_time =1 
    for i in range(circle_time):
        result += cross_validation_experiment(
            miR_tar_matrix, miR_sim*simw, tar_sim*simw, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    print(average_result)

