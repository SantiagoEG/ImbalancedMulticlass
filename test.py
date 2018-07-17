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
          ("ROSboost", imb.ROSboost()),
          ("SMOTEboost", imb.SMOTEBoost()),
          ("RUSboost", imb.RUSBoost()),
          ("TLboost", imb.TLboost())
        ]

for tag, load in datasets:
    dataset = load()
    X = dataset.data
    Y = dataset.target  
        
    
    
    for nclf, clf in algos:
        print '=='*25        
        print "Class order {} for dataset {}\n".format(nclf, tag)
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

'''
==================================================
Class order ROSboost for dataset DIGITS

Fold1
	Acc:0.285714285714
	Training Time:0.568933963776
	Classification Time:0.0115168094635
Fold2
	Acc:0.273480662983
	Training Time:0.565816879272
	Classification Time:0.0114288330078
Fold3
	Acc:0.284122562674
	Training Time:0.546090126038
	Classification Time:0.0114500522614
Fold4
	Acc:0.21568627451
	Training Time:0.560751914978
	Classification Time:0.0113260746002
Fold5
	Acc:0.278873239437
	Training Time:0.569422006607
	Classification Time:0.0113909244537
==================================================
Class order SMOTEboost for dataset DIGITS

Fold1
	Acc:0.285714285714
	Training Time:0.479490995407
	Classification Time:0.011430978775
Fold2
	Acc:0.273480662983
	Training Time:0.483418941498
	Classification Time:0.0114471912384
Fold3
	Acc:0.284122562674
	Training Time:0.483948945999
	Classification Time:0.0114450454712
Fold4
	Acc:0.212885154062
	Training Time:0.491546154022
	Classification Time:0.0112841129303
Fold5
	Acc:0.278873239437
	Training Time:0.494383096695
	Classification Time:0.0114181041718
==================================================
Class order RUSboost for dataset DIGITS

Fold1
	Acc:0.46978021978
	Training Time:0.558105945587
	Classification Time:0.0118148326874
Fold2
	Acc:0.370165745856
	Training Time:0.518282175064
	Classification Time:0.0117061138153
Fold3
	Acc:0.364902506964
	Training Time:0.554261922836
	Classification Time:0.0118181705475
Fold4
	Acc:0.260504201681
	Training Time:0.512488126755
	Classification Time:0.0116209983826
Fold5
	Acc:0.287323943662
	Training Time:0.558358192444
	Classification Time:0.0118429660797
==================================================
Class order TLboost for dataset DIGITS

Fold1
	Acc:0.200549450549
	Training Time:13.9705519676
	Classification Time:0.0116279125214
Fold2
	Acc:0.201657458564
	Training Time:14.0244500637
	Classification Time:0.0115950107574
Fold3
	Acc:0.194986072423
	Training Time:14.0095899105
	Classification Time:0.0116629600525
Fold4
	Acc:0.193277310924
	Training Time:13.9878702164
	Classification Time:0.0114889144897
Fold5
	Acc:0.197183098592
	Training Time:14.0124120712
	Classification Time:0.0115299224854
==================================================
Class order ROSboost for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.216493844986
	Classification Time:0.00423216819763
Fold2
	Acc:0.933333333333
	Training Time:0.215262889862
	Classification Time:0.00423812866211
Fold3
	Acc:0.9
	Training Time:0.215595960617
	Classification Time:0.0042188167572
Fold4
	Acc:0.933333333333
	Training Time:0.215348958969
	Classification Time:0.00424098968506
Fold5
	Acc:1.0
	Training Time:0.215986967087
	Classification Time:0.00422787666321
==================================================
Class order SMOTEboost for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.239869117737
	Classification Time:0.00423192977905
Fold2
	Acc:0.933333333333
	Training Time:0.242223024368
	Classification Time:0.00428509712219
Fold3
	Acc:0.9
	Training Time:0.244747161865
	Classification Time:0.00425291061401
Fold4
	Acc:0.933333333333
	Training Time:0.247266054153
	Classification Time:0.00423002243042
Fold5
	Acc:1.0
	Training Time:0.252019882202
	Classification Time:0.00422120094299
==================================================
Class order RUSboost for dataset IRIS

Fold1
	Acc:0.666666666667
	Training Time:0.229157924652
	Classification Time:0.00421500205994
Fold2
	Acc:0.666666666667
	Training Time:0.230819940567
	Classification Time:0.00425505638123
Fold3
	Acc:0.666666666667
	Training Time:0.229274988174
	Classification Time:0.00422215461731
Fold4
	Acc:0.666666666667
	Training Time:0.231146097183
	Classification Time:0.00420689582825
Fold5
	Acc:0.666666666667
	Training Time:0.229363918304
	Classification Time:0.0042040348053
==================================================
Class order TLboost for dataset IRIS

Fold1
	Acc:0.666666666667
	Training Time:0.290214776993
	Classification Time:0.00428891181946
Fold2
	Acc:0.666666666667
	Training Time:0.326241016388
	Classification Time:0.00421905517578
Fold3
	Acc:0.666666666667
	Training Time:0.288220882416
	Classification Time:0.00421786308289
Fold4
	Acc:0.666666666667
	Training Time:0.288480043411
	Classification Time:0.00422501564026
Fold5
	Acc:0.666666666667
	Training Time:0.289098978043
	Classification Time:0.00422716140747
'''
