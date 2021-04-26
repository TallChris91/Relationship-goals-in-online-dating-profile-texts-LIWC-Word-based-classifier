def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys
import os
import os.path
#sys.path.append(r'C:\\Users\\u1269857\\AppData\\Local\\Continuum\\Anaconda2\\Lib\\site-packages')
sys.path.append('C:\\Program Files\\Anaconda3\\Lib\\site-packages')
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
import regex as re
import pickle
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import warnings

def BayesOptimize(currentpath, featuretype, features, categories, algorithm):
    if not os.path.exists(currentpath + '/OptimalParameters/'):
        os.makedirs(currentpath + '/OptimalParameters/')
    print('Calculating the optimal parameters for ' + featuretype + ' features using the ' + algorithm + ' algorithm')

    def xgcv(min_child_weight, colsample_bytree, max_depth, subsample, gamma, alpha):
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        clf = XGBClassifier(min_child_weight=int(min_child_weight),
                            colsample_bytree=max(min(colsample_bytree, 1), 0),
                            max_depth=int(max_depth),
                            subsample=max(min(subsample, 1), 0),
                            gamma=max(gamma, 0),
                            alpha=max(alpha, 0))

        clf.fit(features, categories)

        # Use 10-fold cross validation, which makes the prediction scores more stable
        num_folds = 10
        # Change random_state around for different combination methods
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
        # Now do predictions for all the files
        predictions = cross_val_predict(clf, features, categories, cv=kfold)
        acc = accuracy_score(categories, predictions)
        return acc

    def xgoptimize():
        xgBO = BayesianOptimization(xgcv, {'min_child_weight': (1, 20),
                                           'colsample_bytree': (0.1, 1),
                                           'max_depth': (5, 15),
                                           'subsample': (0.5, 1),
                                           'gamma': (0, 10),
                                           'alpha': (0, 10)})

        xgBO.maximize()
        return xgBO.res['max']['max_params']

    def rfccv_part(n_estimators, min_samples_split, max_features):
        sample = np.random.choice(np.arange(len(categories)), int(round((len(categories) / 100) * 10, 0)), replace=False)
        newfeatures = features
        if featuretype == 'SuperVector':
            newfeatures = newfeatures.tocsr()
        samplefeatures = newfeatures[sample]
        samplecategories = categories[sample]
        clf = RandomForestClassifier(n_estimators=int(n_estimators),
                                     min_samples_split=int(min_samples_split),
                                     max_features=min(max_features, 0.999))
        clf.fit(samplefeatures, samplecategories)
        # Use 10-fold cross validation, which makes the prediction scores more stable
        num_folds = 10
        # Change random_state around for different combination methods
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
        # Now do predictions for all the files
        predictions = cross_val_predict(clf, samplefeatures, samplecategories, cv=kfold)
        acc = accuracy_score(samplecategories, predictions)
        return acc

    def rfccv_all(n_estimators, min_samples_split, max_depth, min_samples_leaf, max_features):
        clf = RandomForestClassifier(n_estimators=int(n_estimators),
                                     min_samples_split=int(min_samples_split),
                                     max_depth=int(max_depth),
                                     min_samples_leaf=int(min_samples_leaf),
                                     max_features=min(max_features, 0.999))
        clf.fit(features, categories)
        # Use 10-fold cross validation, which makes the prediction scores more stable
        num_folds = 10
        # Change random_state around for different combination methods
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
        # Now do predictions for all the files
        predictions = cross_val_predict(clf, features, categories, cv=kfold)
        acc = accuracy_score(categories, predictions)
        return acc

    def rfcoptimize():
        if (featuretype == 'WordNGram') or (featuretype == 'SuperVector'):
            rfcBO = BayesianOptimization(rfccv_part, {'n_estimators': (10, 250),
                                                      'min_samples_split': (2, 25),
                                                      'max_features': (0.1, 0.999)})
        else:
            rfcBO = BayesianOptimization(rfccv_all, {'n_estimators': (10, 2000),
                                                     'min_samples_split': (2, 25),
                                                     'max_depth': (10, 110),
                                                     'min_samples_leaf': (1, 4),
                                                     'max_features': (0.1, 0.999)})


        rfcBO.maximize()
        return rfcBO.res['max']['max_params']

    def treecv(min_samples_split):
        clf = DecisionTreeClassifier(min_samples_split=int(min_samples_split))
        clf.fit(features, categories)

        # Use 10-fold cross validation, which makes the prediction scores more stable
        num_folds = 10
        # Change random_state around for different combination methods
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
        # Now do predictions for all the files
        predictions = cross_val_predict(clf, features, categories, cv=kfold)
        acc = accuracy_score(categories, predictions)
        return acc

    def treeoptimize():
        treeBO = BayesianOptimization(treecv, {'min_samples_split': (2, 25)})
        treeBO.maximize()
        return treeBO.res['max']['max_params']

    def adacv(n_estimators, learning_rate):
        clf = AdaBoostClassifier(n_estimators=int(n_estimators), learning_rate=learning_rate)
        clf.fit(features, categories)

        # Use 10-fold cross validation, which makes the prediction scores more stable
        num_folds = 10
        # Change random_state around for different combination methods
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
        # Now do predictions for all the files
        predictions = cross_val_predict(clf, features, categories, cv=kfold)
        acc = accuracy_score(categories, predictions)
        return acc

    def adaoptimize():
        adaBO = BayesianOptimization(adacv, {'n_estimators': (10, 250), 'learning_rate': (0.0001, 0.9999)})
        adaBO.maximize()
        return adaBO.res['max']['max_params']

    def nbcv(alpha):
        clf = MultinomialNB(alpha=alpha)
        clf.fit(features, categories)

        # Use 10-fold cross validation, which makes the prediction scores more stable
        num_folds = 10
        # Change random_state around for different combination methods
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
        # Now do predictions for all the files
        predictions = cross_val_predict(clf, features, categories, cv=kfold)
        acc = accuracy_score(categories, predictions)
        return acc

    def nboptimize():
        nbBO = BayesianOptimization(nbcv, {'alpha': (0.001, 10.0)})
        nbBO.maximize()
        return nbBO.res['max']['max_params']

    def svmcv(C):
        n_estimators = 10
        clf = BaggingClassifier(LinearSVC(C=C), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
        clf.fit(features, categories)

        # Use 10-fold cross validation, which makes the prediction scores more stable
        num_folds = 10
        # Change random_state around for different combination methods
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
        # Now do predictions for all the files
        predictions = cross_val_predict(clf, features, categories, cv=kfold)
        acc = accuracy_score(categories, predictions)
        return acc

    def svmoptimize():
        svmBO = BayesianOptimization(svmcv, {'C': (0.001, 10.0)})
        svmBO.maximize()
        return svmBO.res['max']['max_params']

    if algorithm == 'svm':
        optdict = svmoptimize()
    elif algorithm == 'nb':
        optdict = nboptimize()
    elif algorithm == 'tree':
        optdict = treeoptimize()
    elif algorithm == 'ada':
        optdict = adaoptimize()
    elif algorithm == 'rfc':
        optdict = rfcoptimize()
    elif algorithm == 'xg':
        optdict = xgoptimize()

    with open(currentpath + '/OptimalParameters/' + featuretype + algorithm + 'OptdictParams.p', 'wb') as f:
        print('Saving the optimal parameters')
        pickle.dump(optdict, f)