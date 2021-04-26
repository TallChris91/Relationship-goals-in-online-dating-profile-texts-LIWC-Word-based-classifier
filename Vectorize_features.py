import os
import os.path
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import numpy as np
import scipy

def LoadLists(currentpath):
    with open(currentpath + '/FeaturesList/WordNGramsFeatures.p', 'rb') as f:
        print('Loading WordNgrams features')
        wordngramslist = pickle.load(f)
    with open(currentpath + '/FeaturesList/WordNGramsFeaturesTotals.p', 'rb') as f:
        print('Loading WordNgrams total features')
        wordngramstotallist = pickle.load(f)
    with open(currentpath + '/FeaturesList/LIWCFeatures.p', 'rb') as f:
        print('Loading LIWC features')
        liwcfeatureslist = pickle.load(f)
    with open(currentpath + '/FeaturesList/LIWCFeaturesTotals.p', 'rb') as f:
        print('Loading LIWC total features')
        liwctotalfeatureslist = pickle.load(f)
    with open(currentpath + '/FeaturesList/SelectLIWCFeatures.p', 'rb') as f:
        print('Loading LIWC features')
        selectliwcfeatureslist = pickle.load(f)
    with open(currentpath + '/FeaturesList/SelectLIWCFeaturesTotals.p', 'rb') as f:
        print('Loading LIWC total features')
        selectliwctotalfeatureslist = pickle.load(f)
    with open(currentpath + '/FeaturesList/Categories.p', 'rb') as f:
        print('Loading Dutch/Flemish Categories')
        categorieslist = pickle.load(f)

    return wordngramslist, wordngramstotallist, liwcfeatureslist, liwctotalfeatureslist, selectliwcfeatureslist, selectliwctotalfeatureslist, categorieslist

def WordNGramsVectorizer(currentpath, wordngramslist, wordngramstotallist):
    print('Creating WordNGrams Vector')
    #Transform the ngrams into a sparse numpy array which scikit learn can use
    vec = DictVectorizer(sparse=True)
    transvec = vec.fit_transform(wordngramslist)
    features = vec.get_feature_names()

    #Save the features and feature names into a pickle file
    print('Saving WordNGrams Array')
    with open(currentpath + '/Vectorfiles/WordNGramArray.p', 'wb') as f:
        pickle.dump(transvec, f)
    print('Saving WordNGrams Feature List')
    with open(currentpath + '/Vectorfiles/WordNGramFeatures.p', 'wb') as f:
        pickle.dump(features, f)

    transvectotal = vec.fit_transform(wordngramstotallist)
    print('Saving WordNGrams Total Array')
    with open(currentpath + '/Vectorfiles/WordNGramTotalArray.p', 'wb') as f:
        pickle.dump(transvectotal, f)

    return transvec, transvectotal, features

def LIWCVectorizer(currentpath, liwcfeatureslist, liwctotalfeatureslist, selection):
    # Transform the liwc list of dicts into a vector
    print('Creating LIWC Vector')
    vec = DictVectorizer(sparse=False)
    transvec = vec.fit_transform(liwcfeatureslist)
    # And keep the feature names (you never know!)
    features = vec.get_feature_names()
    # Save the features and feature names into a numpy array file
    print('Saving LIWC Array')
    with open(currentpath + '/Vectorfiles/' + selection + 'LIWCArray.npy', 'wb') as f:
        np.save(f, transvec)
    print('Saving LIWC Feature List')
    with open(currentpath + '/Vectorfiles/' + selection + 'LIWCFeatures.npy', 'wb') as f:
        np.save(f, features)

    print('Saving LIWC Total Array')
    transvectotal = vec.fit_transform(liwctotalfeatureslist)
    with open(currentpath + '/Vectorfiles/' + selection + 'LIWCTotalArray.npy', 'wb') as f:
        np.save(f, transvectotal)
    return transvec, transvectotal, features

def supervector(currentpath, vecword, vecwordtotal, vecliwc, vecliwctotal, wordfeats, liwcfeats):
    print('Creating Supervector')
    features = np.concatenate((wordfeats, liwcfeats), axis=0)
    features = features.tolist()
    for idx, val in enumerate(vecliwc):
        if idx == 0:
            transvec = scipy.sparse.hstack((vecword[idx], vecliwc[idx]))
        else:
            combine = scipy.sparse.hstack((vecword[idx], vecliwc[idx]))
            transvec = scipy.sparse.vstack((transvec, combine))
    print('Saving Supervector Array')
    with open(currentpath + '/Vectorfiles/SuperVectorArray.p', 'wb') as f:
        pickle.dump(transvec, f)
    print('Saving Supervector Feature List')
    with open(currentpath + '/Vectorfiles/SuperVectorFeatures.p', 'wb') as f:
        pickle.dump(features, f)

    for idx, val in enumerate(vecliwctotal):
        if idx == 0:
            transvectotal = scipy.sparse.hstack((vecwordtotal[idx], vecliwctotal[idx]))
        else:
            combinetotal = scipy.sparse.hstack((vecwordtotal[idx], vecliwctotal[idx]))
            transvectotal = scipy.sparse.vstack((transvectotal, combinetotal))

    print('Saving Supervector Total Array')
    with open(currentpath + '/Vectorfiles/SuperVectorTotalArray.p', 'wb') as f:
        pickle.dump(transvectotal, f)

    return transvec, transvectotal, features

def CategoriesVectorizer(currentpath, categorieslist):
    print('Creating Categories Vector')
    lb = preprocessing.LabelBinarizer()
    categories = lb.fit_transform(categorieslist)
    features = lb.classes_
    categories = categories.ravel()
    print('Saving Categories Array')
    with open(currentpath + '/Vectorfiles/CategoriesArray.npy', 'wb') as f:
        np.save(f, categories)
    print('Saving Categories Feature List')
    with open(currentpath + '/Vectorfiles/CategoriesFeatures.npy', 'wb') as f:
        np.save(f, features)
    return categories, features

def LoadFiles(currentpath):
    print('Loading WordNGrams Array')
    with open(currentpath + '/Vectorfiles/WordNGramArray.p', 'rb') as f:
        vecword = pickle.load(f)
    with open(currentpath + '/Vectorfiles/WordNGramTotalArray.p', 'rb') as f:
        vecwordtotal = pickle.load(f)
    print('Loading WordNGrams Feature List')
    with open(currentpath + '/Vectorfiles/WordNGramFeatures.p', 'rb') as f:
        wordfeats = pickle.load(f)
    print('Loading LIWC Array')
    with open(currentpath + '/Vectorfiles/LIWCArray.npy', 'rb') as f:
        vecliwc = np.load(f)
    with open(currentpath + '/Vectorfiles/LIWCTotalArray.npy', 'rb') as f:
        vecliwctotal = np.load(f)
    print('Loading LIWC Feature List')
    with open(currentpath + '/Vectorfiles/LIWCFeatures.npy', 'rb') as f:
        liwcfeats = np.load(f)
    print('Loading Selection LIWC Array')
    with open(currentpath + '/Vectorfiles/SelectLIWCArray.npy', 'rb') as f:
        selectvecliwc = np.load(f)
    with open(currentpath + '/Vectorfiles/SelectLIWCTotalArray.npy', 'rb') as f:
        selectvecliwctotal = np.load(f)
    print('Loading Selection LIWC Feature List')
    with open(currentpath + '/Vectorfiles/SelectLIWCFeatures.npy', 'rb') as f:
        selectliwcfeats = np.load(f)
    return vecword, vecwordtotal, vecliwc, vecliwctotal, selectvecliwc, selectvecliwctotal, wordfeats, liwcfeats, selectliwcfeats

def SaveAll():
    currentpath = os.getcwd()

    #Create the Vectorfiles path if it doesn't exist
    if not os.path.exists(currentpath + '/Vectorfiles/'):
        os.makedirs(currentpath + '/Vectorfiles/')

    #Load the lists
    if not os.path.isfile(currentpath + '/Vectorfiles/CategoriesFeatures.npy'):
        wordngramslist, wordngramstotallist, liwcfeatureslist, liwctotalfeatureslist, selectliwcfeatureslist, selectliwctotalfeatureslist, categorieslist = LoadLists(currentpath)

    #Vectorize the WordNGrams features and save them
    if not os.path.isfile(currentpath + '/Vectorfiles/WordNGramFeatures.p'):
        vecword, vecwordtotal, wordfeats = WordNGramsVectorizer(currentpath, wordngramslist, wordngramstotallist)

    # Vectorize the LIWC features and save them
    if not os.path.isfile(currentpath + '/Vectorfiles/LIWCFeatures.p'):
        vecliwc, vecliwctotal, liwcfeats = LIWCVectorizer(currentpath, liwcfeatureslist, liwctotalfeatureslist, '')

    # Vectorize the Selected LIWC features and save them
    if not os.path.isfile(currentpath + '/Vectorfiles/SelectLIWCFeatures.p'):
        selectvecliwc, selectvecliwctotal, selectliwcfeats = LIWCVectorizer(currentpath, selectliwcfeatureslist, selectliwctotalfeatureslist, 'Select')

    #Combine the WordNGrams and LIWC features into one big supervector
    if not os.path.isfile(currentpath + '/Vectorfiles/SuperVectorFeatures.p'):
        vecword, vecwordtotal, vecliwc, vecliwctotal, selectvecliwc, selectvecliwctotal, wordfeats, liwcfeats, selectliwcfeats = LoadFiles(currentpath)
        supervectorvec, supervectorvectotal, supervectorfeatures = supervector(currentpath, vecword, vecwordtotal, vecliwc, vecliwctotal, wordfeats, liwcfeats)
    #Vectorize the categories and save them
    if not os.path.isfile(currentpath + '/Vectorfiles/CategoriesFeatures.npy'):
        CategoriesVectorizer(currentpath, categorieslist)

#SaveAll()