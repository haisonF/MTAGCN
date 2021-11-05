# MTAGCN
**MTAGCN: Predicting miRNA-target associations in Camellia sinensis var.assamica through graph convolution neural network**



## Datasets

* miRNA_sim.csv: miRNA similarity matrix, which is calculated based on miRNA features.
* target_sim.csv: target-similarity matrix, which is calculated based on target features. 
* miRNA_target.csv: the miRNA_target association network, which is calculated based on miRNA mesh descriptors.

  



# Run steps

1. To generate training data and test data.
2.  Run main.py  to train the model and obtain the predicted scores for  miRNA-target associations.

# Requirements

* MTAGCN is implemented to work under Python 3.6. 
* tensorflow==12.20
* numpy==1.19.2
* scipy==1.1.0
* sklearn==0.24.2
* pandas==1.1.5

