# ImbalancedMulticlass: Coping with Class Imbalance in Multiclassification

This module implements Machine Learning algorithms to cope with Class Imbalance in multiclassification. ImbalancedMulticlass includes a METACost implementation and various boosting algorithms that include resampling during the learning. In the 'test.py' script you can found examples of usage. 

Boosting algorithms included:
- SMOTEBoost
- ROSBoost
- TLBoost
- RUSBoost

All of these algorithms have been employed in the scientific article "Exploratory Study on Class Imbalance and Solutions for Network Traffic Classification". In this article you can find the strategies to adjust the ratio between classes during bossting learning and to set the classification costs in the case of METACost.
Please, cite this research if you use this Python module.


