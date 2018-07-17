# -*- coding: utf-8 -*-
"""
Testing
"""

from sklearn.datasets import load_digits, load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

"""
Base Estimator & Datasets
"""
base_estimator = DecisionTreeClassifier()
datasets = [
            ("DIGITS", load_digits),
            ("IRIS", load_iris)
            ]


"""
Algorithms
"""
import imblMulticlass as imb
algos = [
          ("ROSboost", imb.ROSboost),
          ("SMOTEboost", imb.SMOTEBoost),
          ("RUSboost", imb.RUSBoost),
          ("TLboost", imb.TLboost)
        ]

for tag, load in datasets:
    dataset = load()
    X = dataset.data
    Y = dataset.target  
        
    
    for nclf, clf in algos:


        """
        Testing the returned base estimator orders
        """
        for order in order_iter.as_matrix():
            print '=='*25        
            print "Class order {} for dataset {}\n".format(order, tag)
            tec_clf.order = order
            SKFold = StratifiedKFold(n_splits = 5, random_state = 1)
            
            
            i = 1 
            for train_idx, test_idx in SKFold.split(X,Y):
                x_train, y_train = X[train_idx, :], Y[train_idx]
                x_test, y_test = X[test_idx, :], Y[test_idx]
                """
                Training
                """     
                tt = time.time()
                clf.fit(x_train, y_train)
                tt = time.time() - tt
                """
                Validation
                """   
                ct = time.time()
                y_pred = clf.predict(x_test)
                ct = time.time() - ct
                
                acc = accuracy_score(y_test, y_pred)
                
                
                print "Fold{}".format(i)
                print "\tAcc:{}" .format(acc)
                print "\tTraining Time:{}".format(tt)
                print "\tClassification Time:{}".format(ct)
                i = i+1
    
    
