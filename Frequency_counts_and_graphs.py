import numpy as np
import pickle
import os
import operator
import csv
import itertools
import sys
import scipy
import regex as re
import csv

def LIWC_Categories(currentpath):
    liwcdict = {}
    with open(currentpath + '/LIWC-Features/LIWC-Categories.txt', 'r',
              encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t')

        for idx, row in enumerate(reader):
            liwcdict.update({int(row[0]): row[1]})
    return liwcdict

def Best_Algorithms(currentpath):
    featuretypelist = ['WordNGram', 'LIWC', 'SuperVector']

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

    maxlist = []
    for featuretype in featuretypelist:
        #Get the subset containing only the feature types
        typescores = [x for x in newscores if x['Feature type'] == featuretype]
        #And sort by the item with the highest accuracy score
        typemax = sorted(typescores, key=lambda k: k['Accuracy score'])
        #Then convert the list so that it only contains the algorithms sorted from low to high
        algorithm = [v['Algorithm'] for v in typemax]
        #Reverse order (practical for the next function)
        algorithm = algorithm[::-1]
        maxlist.append(algorithm)

    return maxlist

def Get_Important_Features(currentpath, n):
    #Grab the algorithms with the highest accuracy score from the Classifier_scores file using the Best_Algorithms function
    maxlist = Best_Algorithms(currentpath)
    featuretypelist = ['WordNGram', 'LIWC', 'SuperVector']
    totalsorted = []
    for featidx, featuretype in enumerate(featuretypelist):
        # Load the Feature_importances for the highest algorithm (that returned feature importances)
        for algorithmidx, algorithmtype in enumerate(maxlist):
            if os.path.isfile(currentpath + '/Classifierfiles/Feature_importances' + maxlist[featidx][algorithmidx] + featuretype + '.p'):
                with open(currentpath + '/Classifierfiles/Feature_importances' + maxlist[featidx][algorithmidx] + featuretype + '.p', 'rb') as f:
                    importances = pickle.load(f)
                break

        # Load the names of the features
        if featuretype == 'LIWC':
            with open(currentpath + '/Vectorfiles/' + featuretype + 'Features.npy', 'rb') as f:
                feats = np.load(f)
        else:
            with open(currentpath + '/Vectorfiles/' + featuretype + 'Features.p', 'rb') as f:
                feats = pickle.load(f)

        # Make a dictionary first containing the name of the feature and its importance, convert that to a list of tuples sorted by feature importance
        featdict = {}
        for idx, val in enumerate(importances):
            featdict.update({feats[idx]: importances[idx]})
        sorted_featdict = sorted(featdict.items(), key=operator.itemgetter(1))[::-1]

        # Cut off the list at the n-point: giving you the top-n most important features
        # But only cut it off if the n is actually a lower number than the list length
        if n < len(sorted_featdict):
            sorted_featdict = sorted_featdict[:n]
        totalsorted.append(sorted_featdict)

    return totalsorted

def Category_Frequency(currentpath):
    #Load the array and the feature names
    with open(currentpath + '/Vectorfiles/CategoriesArray.npy', 'rb') as f:
        categories = np.load(f)
    with open(currentpath + '/Vectorfiles/CategoriesFeatures.npy', 'rb') as f:
        features = np.load(f)

    #Bincount gets the frequencies for every unique category
    frequency = np.bincount(categories)
    '''
    #Convert this bincount format into a dictionary
    frequencydict = {}
    for idx, val in enumerate(frequency):
        frequencydict.update({features[idx]: frequency[idx]})
    '''
    return frequency

def Get_Ratio(n=10):
    currentpath = os.getcwd()
    #Load the most important features per feature type
    sortedlist = Get_Important_Features(currentpath, n)
    #Load the frequencies date and long term texts (important to get averages)
    frequency = Category_Frequency(currentpath)
    #Load the list with a category label per instance
    with open(currentpath + '/Vectorfiles/CategoriesArray.npy', 'rb') as f:
        categories = np.load(f)

    featuretypelist = ['WordNGram', 'LIWC', 'SuperVector']
    #Iterate over the feature types
    for idx, featuretype in enumerate(featuretypelist):
        #Load the list with the features for every feature type per instance
        if featuretype == 'LIWC':
            with open(currentpath + '/Vectorfiles/' + featuretype + 'TotalArray.npy', 'rb') as f:
                instances = np.load(f)
        else:
            with open(currentpath + '/Vectorfiles/' + featuretype + 'TotalArray.p', 'rb') as f:
                instances = pickle.load(f)

        #Coo format (the format that the supervector uses) is difficult with indexes, but the csr format works, so convert the coo to csr
        if featuretype == 'SuperVector':
            instances = instances.tocsr()
        #Now let's seperate the feature info based on if the text is a date or long term text
        if featuretype == 'LIWC':
            DateTextslist = [x for instanceidx, x in enumerate(instances) if categories[instanceidx] == 0]
            LongTermlist = [x for instanceidx, x in enumerate(instances) if categories[instanceidx] == 1]

            # And sum the values in the lists
            DateTextslist = [sum(i) for i in zip(*DateTextslist)]
            LongTermlist = [sum(i) for i in zip(*LongTermlist)]

        if (featuretype == 'WordNGram') or (featuretype == 'SuperVector'):
            # The WordNGram and the SuperVector are sparse arrays, so they require a different approach
            DateTextslist = None
            LongTermlist = None
            for catidx, val in enumerate(categories):
                #Iterate over the categories list
                if val == 0:
                    #And if it is a DateTexts text, add the instances row with the same index to the DateTextslist
                    if DateTextslist == None:
                        DateTextslist = instances[catidx, :]
                    else:
                        DateTextslist = scipy.sparse.vstack((DateTextslist, instances[catidx, :]))
                if val == 1:
                    #If it is a long term text, add the instances row with the same index to the LongTermlist
                    if LongTermlist == None:
                        LongTermlist = instances[catidx, :]
                    else:
                        LongTermlist = scipy.sparse.vstack((LongTermlist, instances[catidx, :]))

            #Sum the rows and convert the rows to a list
            DateTextslist = DateTextslist.sum(axis=0).tolist()[0]
            LongTermlist = LongTermlist.sum(axis=0).tolist()[0]
        #And calculate the ratios by dividing the summed values by the amount of instances of the DateTexts/long term type
        DateTextsratio = [x / frequency[0] for x in DateTextslist]
        LongTermratio = [x / frequency[1] for x in LongTermlist]
        #Now we have all the frequencies and ratios from all the features, but we only want the frequencies and ratios from the top n most important features
        #So let's focus on that!
        #First load the feature names for the feature type
        if featuretype == 'LIWC':
            with open(currentpath + '/Vectorfiles/' + featuretype + 'Features.npy', 'rb') as f:
                features = np.load(f)
        else:
            with open(currentpath + '/Vectorfiles/' + featuretype + 'Features.p', 'rb') as f:
                features = pickle.load(f)

        featuretypescores = []
        #Get the names of the most important features from the sortedlist
        for important_feature in sortedlist[idx]:
            #Grab the name of the important feature
            important_featurename, important_featurescore = important_feature
            #And get the index number for the DateTextslist/ratio and LongTermlist/ratio
            try:
                feature_index = features.tolist().index(important_featurename)
            except AttributeError:
                feature_index = features.index(important_featurename)
            DateTextslistscore = DateTextslist[feature_index]
            LongTermlistscore = LongTermlist[feature_index]
            DateTextslistratio = DateTextsratio[feature_index]
            LongTermlistratio = LongTermratio[feature_index]
            #Finally append all this info in tuple form to a list that collects all the informatio for the feature type
            featuretypescores.append([important_featurename, important_featurescore, DateTextslistscore, LongTermlistscore, DateTextslistratio, LongTermlistratio])

        #Convert the LIWC featurenames to their original names instead of the numbers
        if (featuretype == 'LIWC') or (featuretype == 'SuperVector'):
            #Get the categories in a dictionary form
            liwcdict = LIWC_Categories(currentpath)
            for ftidx, val in enumerate(featuretypescores):
                #Get the featurename
                featurename = featuretypescores[ftidx][0]
                #If it is a LIWC feature
                if re.search(r"^liwc\d+$", featurename):
                    #Retrieve the category number
                    catnumber = int(re.sub(r'^liwc', '', featurename))
                    #And replace the features with names like 'liwc12' with the actual names of the LIWC category
                    featuretypescores[ftidx][0] = liwcdict[catnumber] + ' (LIWC)'

        if not os.path.exists(currentpath + '/Feature_scores'):
            os.makedirs(currentpath + '/Feature_scores')
        print('Writing csv file')
        with open(currentpath + '/Feature_scores/Most_important_features' + featuretype + '.csv', 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['Feature name', 'Importance score', 'DateTexts total', 'LongTerm total', 'DateTexts average', 'LongTerm average'])
            for row in featuretypescores:
                csvwriter.writerow(row)

#Get_Ratio(n=20)