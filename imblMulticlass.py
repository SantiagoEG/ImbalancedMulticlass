# -*- coding: utf-8 -*-
"""
imblMulticlass includes the following algorithms:

- ROSBoost
- SMOTEBoost
- RUSBoost
- TLBoost
- METACost

All of these algorithms have been employed in the scientific article [1]. In 
this article you can find the strategies to adjust the ratio between classes 
during bossting learning and to set the classification costs in the case of 
METACost. 

Please, cite this research if you use this module ([1]).

References
----------
   [1] Santiago E. Gómez..., (In production)
   "Exploratory Study on Class Imbalance and Solutions for Network Traffic Classification"
   Neurocomputing, 2018

   [2] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
       "SMOTEBoost: Improving Prediction of the Minority Class in
       Boosting." European Conference on Principles of Data Mining and
       Knowledge Discovery (PKDD), 2003.
   [3] N. V. Chawla, A. Lazarevic, L. O. Hall, and K. W. Bowyer.
       "SMOTEBoost: Improving Prediction of the Minority Class in
       Boosting." European Conference on Principles of Data Mining and
       Knowledge Discovery (PKDD), 2003.       
   [4] C. Seiffert, T. M. Khoshgoftaar, J. V. Hulse, and A. Napolitano.
       "RUSBoost: Improving Classification Performance when Training Data
       is Skewed". International Conference on Pattern Recognition
       (ICPR), 2008.       
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

"""
MODULES
"""
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y, check_array
from sklearn.utils import shuffle



"""
ROSBOOST
"""
class ROSboost(AdaBoostClassifier):
    """Implementation of ROSBoost.

    ROSBoost introduces data sampling into the AdaBoost algorithm by randomly
    oversampling the minority classes on each boosting iteration [2].

    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        Number of new synthetic samples per boosting step.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.algorithm = algorithm
        self.sampler = RandomOverSampler(random_state=random_state)

        super(ROSboost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)


    def ratio_OS(self, y, n = 5):
        """ Create a dictionaries to adjust ratios between classes for boosting
        learning. 
        
        Parameters
        ----------
        n : number of boosting iterations
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).      

        Returns
        -------
        ratios: a list with ratios for each boosting iteration.
        """
        counts = np.bincount(y)
        n_samples = int(np.max(counts[counts > 0]))
        os_classes = np.where((counts < n_samples) & (counts > 0))[0]
        in_per_class = np.zeros(shape = counts.shape)
        in_per_class[counts > 0] = (n_samples - counts[counts > 0])/n
        ratios = []
        for j in range(n):
            ratio = {}    
            for i in os_classes:
                ratio[i] = counts[i]+int(in_per_class[i]*(j+1))
            ratios.append(ratio)
        return ratios


    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing SMOTE during each boosting step.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        ratios = self.ratio_OS(y, self.n_estimators)
        """BOOSTING LEARNING"""
        for iboost in range(self.n_estimators):
            l = y.shape[0]
            """RESAMPLING"""
            self.sampler.ratio = ratios[iboost]
            X, y = self.sampler.fit_sample(X, y)
            
            if iboost != 0:
                sample_weight_syn = np.ones(shape = (y.shape[0]-l,))
                for cls in ratios[iboost].keys():
                    idx_c = y[:l] == cls
                    idx_c_sync = y[l:] == cls
                    sample_weight_syn[idx_c_sync] = np.min(sample_weight[idx_c])
            else:
                sample_weight_syn = np.min(sample_weight)*np.ones(shape = (y.shape[0]-l,))
    
            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))
                
            X, y, sample_weight = shuffle(X, y, sample_weight,
                                          random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self


"""
SMOTEBOOST
"""
class SMOTEBoost(AdaBoostClassifier):
    """Implementation of SMOTEBoost.

    SMOTEBoost introduces data sampling into the AdaBoost algorithm by
    oversampling the minority class using SMOTE on each boosting iteration [3].

    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.

    Parameters
    ----------
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self,
                 k_neighbors=2,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.algorithm = algorithm
        self.sampler = SMOTE(k_neighbors=k_neighbors,
                           random_state=random_state)

        super(SMOTEBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)


    def ratio_OS(self, y, n = 5):
        """ Create a dictionaries to adjust ratios between classes for boosting
        learning. 
        
        Parameters
        ----------
        n : number of boosting iterations
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).      

        Returns
        -------
        ratios: a list with ratios for each boosting iteration.
        """
        counts = np.bincount(y)
        n_samples = int(np.max(counts[counts > 0]))
        os_classes = np.where((counts < n_samples) & (counts > 0))[0]
        in_per_class = np.zeros(shape = counts.shape)
        in_per_class[counts > 0] = (n_samples - counts[counts > 0])/n
        ratios = []
        for j in range(n):
            ratio = {}    
            for i in os_classes:
                ratio[i] = counts[i]+int(in_per_class[i]*(j+1))
            ratios.append(ratio)
        return ratios


    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing SMOTE during each boosting step.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")


        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        ratios = self.ratio_OS(y, self.n_estimators)
        """BOOSTING LEARNING"""
        for iboost in range(self.n_estimators):
            l = y.shape[0]
            """RESAMPLING"""
            self.sampler.ratio = ratios[iboost]
            X, y = self.sampler.fit_sample(X, y)
            
            
            if iboost != 0:
                sample_weight_syn = np.ones(shape = (y.shape[0]-l,))
                for cls in ratios[iboost].keys():
                    idx_c = y[:l] == cls
                    idx_c_sync = y[l:] == cls
                    sample_weight_syn[idx_c_sync] = np.min(sample_weight[idx_c])
            else:
                sample_weight_syn = np.min(sample_weight)*np.ones(shape = (y.shape[0]-l,))
                
            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))
                
            X, y, sample_weight = shuffle(X, y, sample_weight,
                                          random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

"""
RUSBOOST
"""
class RUSBoost(AdaBoostClassifier):
    """Implementation of RUSBoost.

    RUSBoost introduces data sampling into the AdaBoost algorithm by
    undersampling the majority class using random undersampling (with or
    without replacement) on each boosting iteration [4].

    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.

    Parameters
    ----------
    with_replacement : bool, optional (default=True)
        Undersample with replacement.
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self,
                 with_replacement=True,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.algorithm = algorithm
        self.sampler = RandomUnderSampler(random_state=10, ratio = 'auto',
                                          return_indices = True, replacement = with_replacement)

        super(RUSBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def ratio_US(self, y, n = 5):
        """ Create a dictionaries to adjust ratios between classes for boosting
        learning. 
        
        Parameters
        ----------
        n : number of boosting iterations
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).      

        Returns
        -------
        ratios: a list with ratios for each boosting iteration.
        """
        counts = np.bincount(y)
        n_samples = int(np.mean(counts[counts > 0]))
        us_classes = np.where(counts > n_samples)[0]
        in_per_class = (counts - n_samples)/n
        in_per_class[in_per_class < 0] = 0
        ratios = []
        for j in range(n):
            ratio = {} 
            for i in us_classes:
                if counts[i]-int(counts[i]*.1) < np.min(counts):
                    ratio[i] = np.min(counts)
                else:
                    ratio[i] = counts[i]-int(in_per_class[i]*(j+1))
            ratios.append(ratio)
        return ratios

                

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        random_state = check_random_state(self.random_state)
        ratios = self.ratio_US(y, self.n_estimators)
        """BOOSTING LEARNING"""
        for iboost in range(self.n_estimators):
            """RESAMPLING"""
            self.sampler.ratio = ratios[iboost]
            X, y, idx = self.sampler.fit_sample(X, y)
            sample_weight = sample_weight[idx]
            sample_weight = np.squeeze(normalize(sample_weight, axis=0, norm='l1'))
            X, y, sample_weight = shuffle(X, y, sample_weight, random_state = 10)
            

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self
        
"""
TLBOOST
"""
class TLboost(AdaBoostClassifier):
    """Implementation of TLBoost.

    TLBoost introduces data sampling into the AdaBoost algorithm by
    undersampling the majority classes via removing Tomek Links on each 
    boosting iteration [1].

    This implementation inherits methods from the scikit-learn 
    AdaBoostClassifier class, only modifying the `fit` method.

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.
    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.algorithm = algorithm
        self.sampler = TomekLinks(random_state=10, ratio = 'auto',
                                          return_indices = True)

        super(TLboost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def ratio_US(self, y, n = 5):
        """ Create a dictionaries to adjust ratios between classes for boosting
        learning. 
        
        Parameters
        ----------
        n : number of boosting iterations
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).      

        Returns
        -------
        ratios: a list with ratios for each boosting iteration.
        """
        counts = np.bincount(y)
        n_samples = int(np.mean(counts[counts > 0]))
        us_classes = np.where(counts > n_samples)[0]
        in_per_class = (counts - n_samples)/n
        in_per_class[in_per_class < 0] = 0
        ratios = []
        for j in range(n):
            ratio = {} 
            for i in us_classes:
                if counts[i]-int(counts[i]*.1) < np.min(counts):
                    ratio[i] = np.min(counts)
                else:
                    ratio[i] = counts[i]-int(in_per_class[i]*(j+1))
            ratios.append(ratio)
        return ratios

                

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y),
        performing random undersampling during each boosting step.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.
        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
            
        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        Based on the scikit-learn v0.18 AdaBoostClassifier and
        BaseWeightBoosting `fit` methods.
        """
        # Check that algorithm is supported.
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        ratios = self.ratio_US(y, self.n_estimators)
        """BOOSTING LEARNING"""
        for iboost in range(self.n_estimators):
            """RESAMPLING"""
            self.sampler.ratio = ratios[iboost]
            X, y, idx = self.sampler.fit_sample(X, y)
            sample_weight = sample_weight[idx]
            sample_weight = np.squeeze(normalize(sample_weight, axis=0, norm='l1'))
            X, y, sample_weight = shuffle(X, y, sample_weight, random_state = 10)
            
            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self







