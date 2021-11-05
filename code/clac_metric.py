import numpy as np
from sklearn.metrics import auc ,roc_auc_score,precision_recall_curve,accuracy_score,f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve,precision_score,precision_recall_curve
import random
import scipy.sparse as sp


def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix,test_arr,label):
    test_index = np.where(train_matrix == 0)  
    real_score1 = interaction_matrix[test_index] 
    one=np.ones((len(label),1))
    label=np.hstack((label,one))
    label_neg=generate_test_index(label,len(test_arr)) 
    real_score,predict_score=test_label(predict_matrix,label,test_arr,label_neg)
    return get_metrics(real_score, predict_score)


def get_metrics(y_true,y_scores):
    fpr,tpr,thresholds=roc_curve(y_true,y_scores)
    auc_score=auc(fpr,tpr)
    precison, recall, _thresholds =precision_recall_curve(y_true, y_scores)
    aupr=auc(recall,precison)
    y_scores=np.array(y_scores)
    predict_value=np.where(y_scores>0.5,1,0)
    accuracy=accuracy_score(y_true,predict_value)
    recall=recall_score(y_true,predict_value)
    precision1=precision_score(y_true,predict_value)
    F1_score=f1_score(y_true,predict_value)
    TP = predict_value.dot(y_true.T)
    FP = predict_value.sum()-TP
    FN = y_true.sum()-TP
    TN = len(y_true.T)-TP-FP-FN
    specificity=TN/(TN+FP)
    return [aupr,auc_score,F1_score,accuracy,recall,specificity,precision1]
    




def generate_test_index(label,N):
    A = sp.csr_matrix((label[:,2],(label[:,0]-1, label[:,1]-1)),shape=(290,2876)).toarray()
    num = 0
    mask = np.zeros(A.shape)
    x=[]
    y=[]
    test_neg=()
    while(num<N):
        a = random.randint(0,289)
        b = random.randint(0,2875)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            x.append(a)
            y.append(b)
            num += 1
    test_neg=(x,y)
    return test_neg


def test_label(predict_y_proba,labels,test_arr, test_neg_index): 
    x=[]
    y=[]
    for i in list(labels[test_arr,0]-1):
        x.append(int(i))
    for j in list(labels[test_arr,1]-1):
        y.append(int(j))  
    for i in range(len(test_arr)):
        predict_pos = predict_y_proba[x,y]
    for i in range(len(test_neg_index[0])):
        predict_neg = predict_y_proba[test_neg_index] 
    predict_score = np.hstack((predict_pos,predict_neg))
    pos_labels = np.ones(len(test_arr))
    neg_labels = np.zeros(len(test_neg_index[0]))
    test_labels = np.hstack((pos_labels,neg_labels))
    return test_labels,predict_score




