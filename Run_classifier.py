import sys
sys.path.append('C:/Program Files/Anaconda3/Lib/site-packages')
import os
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pickle
from Hyperparameter_Optimization import BayesOptimize
import warnings
import random

def traintestsplit(features, categories, featuretype, algorithm, currentpath):
    #Start with an empty array to which we will add all the feature importance scores
    combinedfeatureimportances = np.zeros_like(categories[0])

    # Use 10-fold cross validation, which makes the prediction scores more stable
    num_folds = 10
    # Change random_state around for different combination methods
    #Get a list of 10 different random seeds (4294967295 is the maximum that can be assigned)
    randomlist = []
    while len(randomlist) < 10:
        randstate = random.randint(0, 4294967295)
        if randstate not in randomlist:
            randomlist.append(randstate)

    #Go over every random seed
    for randstate in randomlist:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=randstate)
        #And perform 10-fold cross-validation
        for train, test in kfold.split(categories):
            #Assign the train and test indexes to the features and categories
            ytrain = categories[train]
            xtrain = features[train]
            ytest = categories[test]
            xtest = features[test]
            #Get the predictions
            warnings.filterwarnings(action='ignore', category=DeprecationWarning)
            clf = algorithms(algorithm, featuretype, currentpath)
            pred = clf.fit(xtrain, ytrain)

            #Get the feature importances for this fold
            important_features = clf.feature_importances_

            #And combine it with previous folds
            combinedfeatureimportances = combinedfeatureimportances + important_features

    #Normalize the array
    dividedfeatureimportances = [float(i)/sum(combinedfeatureimportances) for i in combinedfeatureimportances]

    return dividedfeatureimportances

def algorithms(inp, featuretype, currentpath):
    with open(currentpath + '/OptimalParameters/' + featuretype + inp + 'OptdictParams.p', 'rb') as f:
        optdict = pickle.load(f)
    if inp == 'svm':
        n_estimators = 10
        return BaggingClassifier(LinearSVC(C=optdict['C']), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
    if inp == 'nb':
        return MultinomialNB(alpha=optdict['alpha'])
    if inp == 'tree':
        return DecisionTreeClassifier(min_samples_split=int(optdict['min_samples_split']))
    if inp == 'ada':
        return AdaBoostClassifier(n_estimators=int(optdict['n_estimators']), learning_rate=optdict['learning_rate'])
    if inp == 'rfc':
        if (featuretype == 'WordNGram') or (featuretype == 'SuperVector'):
            return RandomForestClassifier(n_estimators=int(optdict['n_estimators']),
                                          min_samples_split=int(optdict['min_samples_split']),
                                          max_features=optdict['max_features'])
        else:
            return RandomForestClassifier(n_estimators=int(optdict['n_estimators']),
                                          min_samples_split=int(optdict['min_samples_split']),
                                          max_depth=int(optdict['max_depth']),
                                          min_samples_leaf=int(optdict['min_samples_leaf']),
                                          max_features=optdict['max_features'])
    if inp == 'xg':
        return XGBClassifier(min_child_weight=int(optdict['min_child_weight']),
                             colsample_bytree=optdict['colsample_bytree'],
                             max_depth=int(optdict['max_depth']),
                             subsample=optdict['subsample'],
                             gamma=optdict['gamma'],
                             alpha=optdict['alpha'])

def TrainClassifier(currentpath, features, categories, featuretype, algorithm):
    # Create the Vectorfiles path if it doesn't exist
    if not os.path.exists(currentpath + '/Classifierfiles/'):
        os.makedirs(currentpath + '/Classifierfiles/')

    #Only train the model if no prediction file exists yet
    if not os.path.isfile(currentpath + '/Classifierfiles/Classifier_predictions' + algorithm + featuretype + '.npy'):
        #I use Support Vector Machines for classification here, which is generally a pretty good algorithm, but there are plenty of others to try
        clf = algorithms(algorithm, featuretype, currentpath)
        print('Fitting the model. Algorithm: ' + algorithm + ' Featuretype: ' + featuretype)
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        clf.fit(features, categories)
        #Save the fitted model
        print('Saving the fitted model')
        with open(currentpath + '/Classifierfiles/Classifier_model' + algorithm + featuretype + '.p', 'wb') as f:
            pickle.dump(clf, f)

        #Use 10-fold cross validation, which makes the prediction scores more stable
        num_folds = 10
        #Change random_state around for different combination methods
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

        if (algorithm == 'tree') or (algorithm == 'ada') or (algorithm == 'xg'):
            print('Obtaining the feature importances')
            if featuretype == 'SuperVector':
                features2 = features.tocsr()
                important_features = traintestsplit(features2, categories, featuretype, algorithm, currentpath)
            else:
                important_features = traintestsplit(features, categories, featuretype, algorithm, currentpath)
            print('Saving the feature importances')
            with open(currentpath + '/Classifierfiles/Feature_importances' + algorithm + featuretype + '.p', 'wb') as f:
                pickle.dump(important_features, f)
        '''
        if algorithm == 'rfc':
            print('Obtaining the permutation importances')
            rfC = RandomForestClassifier(n_estimators=100, oob_score=True)
            if (featuretype == 'WordNGram') or (featuretype == 'SuperVector'):
                with open(currentpath + '/Vectorfiles/' + featuretype + 'ArrayLimited.p', 'rb') as f:
                    lil_features = pickle.load(f)
                lil_features = lil_features.tolil()
                rfC.fit(lil_features, categories)
            else:
                rfC.fit(features, categories)
            oobC = RFFeatureImportance.PermutationImportance()
            if (featuretype == 'WordNGram') or (featuretype == 'SuperVector'):
                permutation_features = oobC.featureImportances(rfC, lil_features, categories, 10)
            else:
                permutation_features = oobC.featureImportances(rfC, features, categories, 10)
            print('Saving the permutation importances')
            with open(currentpath + '/Classifierfiles/Permutation_importances' + algorithm + featuretype + '.p', 'wb') as f:
                pickle.dump(permutation_features, f)
        '''
        print('Doing predictions')
        #Now do predictions for all the files
        predictions = cross_val_predict(clf, features, categories, cv=kfold)
        print('Saving prediction array')
        #Save the predictions
        with open(currentpath + '/Classifierfiles/Classifier_predictions' + algorithm + featuretype + '.npy', 'wb') as f:
            np.save(f, predictions)
    # Only get the probabilities if the probability file does not exist yet
    if not os.path.isfile(currentpath + '/Classifierfiles/Classifier_probabilities' + algorithm + featuretype + '.npy'):
        print('Getting the probabilities')
        try:
            probabilities = cross_val_predict(clf, features, categories, cv=kfold, method='predict_proba')
        except UnboundLocalError:
            with open(currentpath + '/Classifierfiles/Classifier_model' + algorithm + featuretype + '.p', 'rb') as f:
                clf = pickle.load(f)
            # Use 10-fold cross validation, which makes the prediction scores more stable
            num_folds = 10
            # Change random_state around for different combination methods
            kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
            probabilities = cross_val_predict(clf, features, categories, cv=kfold, method='predict_proba')
        print('Saving probability array')
        # Save the probabilities
        with open(currentpath + '/Classifierfiles/Classifier_probabilities' + algorithm + featuretype + '.npy', 'wb') as f:
            np.save(f, probabilities)

        # And get the f1, precision, recall and accuracy
        with open(currentpath + '/Classifierfiles/Classifier_predictions' + algorithm + featuretype + '.npy', 'rb') as f:
            predictions = np.load(f)
        f1 = f1_score(categories, predictions)
        precision = precision_score(categories, predictions)
        recall = recall_score(categories, predictions)
        accuracy = accuracy_score(categories, predictions)

        scoretext = 'F1 score: ' + str(f1)
        scoretext += '\n'
        scoretext += 'Precision score: ' + str(precision)
        scoretext += '\n'
        scoretext += 'Recall score: ' + str(recall)
        scoretext += '\n'
        scoretext += 'Accuracy score: ' + str(accuracy)
        scoretext += '\n'
        scoretext += 'Feature type: ' + featuretype
        scoretext += '\n'
        scoretext += 'Algorithm: ' + algorithm
        print(scoretext)
        try:
            with open(currentpath + '/Classifierfiles/Classifier_scores.txt', 'rb') as f:
                scoretextold = f.read()

            scoretextold = scoretextold.decode('utf-8')

            scoretextnew = scoretextold + '\n\n' + scoretext
        except FileNotFoundError:
            scoretextnew = scoretext
        #Save the obtained scores to the Classifier_scores.txt file
        print('Saving classifier scores')
        with open(currentpath + '/Classifierfiles/Classifier_scores.txt', 'wb') as f:
            f.write(bytes(scoretextnew, 'UTF-8'))

def Best_Algorithms(currentpath):
    with open(currentpath + '/Classifierfiles/Classifier_scores.txt', 'rb') as f:
        scores = f.read().decode('utf-8')

    #Convert the Classifier scores text file to a list of dictionaries
    #First split the separate algorithm/feature type combinations
    scores = scores.split('\n\n')
    newscores = []
    for idx, val in enumerate(scores):
        scoredict = {}
        #Split every line containing some information
        scores[idx] = scores[idx].split('\n')
        for score in scores[idx]:
            #And finally split at the colon, and converting it into a dictionary
            score = score.split(': ')
            scoredict.update({score[0]: score[1]})
        newscores.append(scoredict)

    #Get the subset containing only the WordNGram feature types
    wngscores = [x for x in newscores if x['Feature type'] == 'WordNGram']
    #And return the item with the highest accuracy score
    wngmax = max(wngscores, key=lambda x:x['Accuracy score'])

    #Get the subset containing only the syntactical feature types
    liwcscores = [x for x in newscores if x['Feature type'] == 'LIWC']
    #And return the item with the highest accuracy score
    liwcmax = max(liwcscores, key=lambda x:x['Accuracy score'])

    return wngmax['Algorithm'], liwcmax['Algorithm']

def Combine_probabilities(currentpath):
    #Grab the algorithms with the highest accuracy score from the Classifier_scores file using the Best_Algorithms function
    wngmax, liwcmax = Best_Algorithms(currentpath)
    #Combine the prediction probabilities per instance classified using WordNGram and LIWC features
    with open(currentpath + '/Classifierfiles/Classifier_probabilities' + wngmax + 'WordNGram.npy', 'rb') as f:
        wordprobs = np.load(f)
    with open(currentpath + '/Classifierfiles/Classifier_probabilities' + liwcmax + 'LIWC.npy', 'rb') as f:
        liwcprobs = np.load(f)
    print('Preparing the Meta Classifier Vector')
    concat = np.hstack((wordprobs, liwcprobs))
    print('Saving the Meta Classifier Vector')
    with open(currentpath + '/Vectorfiles/FeatCombinedList.npy', 'wb') as f:
        np.save(f, concat)

def RunAll():
    currentpath = os.getcwd()

    featuretypelist = ['WordNGram', 'LIWC', 'SuperVector', 'Meta']
    algorithmlist = ['svm', 'nb', 'tree', 'ada', 'rfc', 'xg']

    for featuretype in featuretypelist:
        with open(currentpath + '/Vectorfiles/CategoriesArray.npy', 'rb') as f:
            categories = np.load(f)
        if (featuretype == 'LIWC'):
            with open(currentpath + '/Vectorfiles/' + featuretype + 'Array.npy', 'rb') as f:
                features = np.load(f)
        if (featuretype == 'WordNGram') or (featuretype == 'SuperVector'):
            with open(currentpath + '/Vectorfiles/' + featuretype + 'Array.p', 'rb') as f:
                features = pickle.load(f)
        if featuretype == 'Meta':
            if not os.path.isfile(currentpath + '/Vectorfiles/FeatCombinedList.npy'):
                Combine_probabilities(currentpath)
            with open(currentpath + '/Vectorfiles/FeatCombinedList.npy', 'rb') as f:
                features = np.load(f)
        for algorithm in algorithmlist:
            #print('Training a classifier on ' + featuretype + ' features with ' + algorithm)
            if not os.path.isfile(currentpath + '/OptimalParameters/' + featuretype + algorithm + 'OptdictParams.p'):
                BayesOptimize(currentpath, featuretype, features, categories, algorithm)
            TrainClassifier(currentpath, features, categories, featuretype, algorithm)

#RunAll()