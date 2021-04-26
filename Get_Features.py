import os
import os.path
import regex as re
from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams
from collections import Counter
import pickle
import sys
import csv
from unidecode import unidecode

def LIWCdict(currentpath):
    liwcdict = {}

    with open(currentpath + '/LIWC-Features/LIWC-Compleet_WithCategories_WithAddedWordsAttractiveness.txt', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for idx, row in enumerate(reader):
            if idx > 0:
                newrow = list(filter(None, row))
                liwcdict.update({newrow[0]: newrow[1:]})

    liwcdictnodiacritics = {unidecode(k): v for k, v in liwcdict.items()}
    return liwcdict, liwcdictnodiacritics

def WordList(text, desiredoutput):
    stopwords = ['aan', 'af', 'al', 'alles', 'als', 'altijd', 'andere', 'ben', 'bij', 'daar', 'dan', 'dat', 'de', 'der',
                 'deze', 'die', 'dit', 'doch', 'doen', 'door', 'dus', 'een', 'eens', 'en', 'er', 'ge', 'geen', 'geweest',
                 'had', 'heb', 'hebben', 'heeft', 'het', 'hier', 'hoe', 'iemand', 'iets', 'in', 'is', 'ja', 'kan', 'kon',
                 'kunnen', 'maar', 'me', 'meer', 'met', 'moet', 'na', 'naar', 'niet', 'niets', 'nog', 'nu', 'of', 'om',
                 'omdat', 'ook', 'op', 'over', 'reeds', 'te', 'tegen', 'toch', 'toen', 'tot', 'uit', 'van', 'veel', 'voor',
                 'want', 'waren', 'was', 'wat', 'wel', 'werd', 'wezen', 'wie', 'wil', 'worden', 'zal', 'zei', 'zich', 'zijn',
                 'zo', 'zonder', 'zou']
    if desiredoutput == 'words':
        sentencelist = []
        for dict in text:
            lemma = dict['lemma'][0]
            sentencelist.append(lemma)

        text2 = []
        text3 = []
        punct = []
        #Loop over all the words in the sentencelist
        for word in sentencelist:
            # If the word is not actually punctuation...
            if re.search(r"^\p{P}+$", word) == None:
                # ... and if the word is not a stop word
                if word.lower() not in stopwords:
                    #... add it to the new text list (which is useful for word n-gram features)
                    text2.append(word)
                # Stopwordness doesn't matter for other features, add all words regardless of being a stop word for these features
                text3.append(word)
            #And in some cases you want the punctuation
            else:
                punct.append(word)
        return text2, text3, punct
    if desiredoutput == 'pos':
        poslist = []
        posclasslist = []
        for dict in text:
            pos = dict['poshead']
            poslist.append(pos)
            posclass = dict['posclass'][0]
            posclasslist.append(posclass)

        return poslist, posclasslist

    if desiredoutput == 'raw':
        #For LIWC-matches we want (preferably) the unlemmatized words and, if they are not found in LIWC, the lemmatized words
        wordlist = []
        lemmalist = []
        for dict in text:
            lemma = dict['lemma'][0]
            word = dict['t']
            # If the word is not actually punctuation...
            if re.search(r"^\p{P}+$", word) == None:
                lemmalist.append(lemma)
                wordlist.append(word)
        return wordlist, lemmalist

def getsentences(line):
    wordlist = []
    for dict in line:
        word = dict['t']
        wordlist.append(word)

    wordstring = ' '.join(wordlist)
    sentences = sent_tokenize(wordstring)
    return sentences

def WordNGrams(line, strip):
    if strip == 'y':
        # Get the text without punctuation and stopwords
        wordlist, allwords, punct = WordList(line, 'words')
    else:
        wordlist = line
        allwords = line

    # This collects all two-word pairs
    bigrams = ngrams(wordlist, 2)
    # And creates a dict containing how often these two-word pairs occur
    bigramscounter = Counter(bigrams)
    bigramscounter = dict(bigramscounter)

    # Do the same for the single words
    unigrams = ngrams(wordlist, 1)
    unigramscounter = Counter(unigrams)
    unigramscounter = dict(unigramscounter)

    # Make a combined list of the bigrams and unigrams
    combine = unigramscounter.copy()
    combine.update(bigramscounter)
    newcombine = {}
    # The keys in the list are stored as tuples. Let's make them strings
    for i in combine:
        newi = list(i)
        newi = " ".join(newi)
        newcombine.update({newi: float(combine[i])})

    newcombinenew = {k: float(v) / float(len(allwords)) for k, v in newcombine.items()}
    return newcombine, newcombinenew

def LIWCFeatures(line, liwcdict, liwcdictnodiacritics):
    wordlist, lemmalist = WordList(line, 'raw')
    liwctotals = {}
    #Check if the regular word is in LIWC
    for idx, word in enumerate(wordlist):
        liwcvalue = None
        if word.lower() in liwcdict:
            liwcvalue = liwcdict[word.lower()]
        #Key is not present
        elif word.lower() in liwcdictnodiacritics:
            #Check if people maybe use the word without diacritics (e.g. beinvloeden instead of beïnvloeden)
            liwcvalue = liwcdictnodiacritics[word.lower()]
        elif lemmalist[idx].lower() in liwcdict:
            #Check if the lemma is in LIWC
            liwcvalue = liwcdict[lemmalist[idx].lower()]
        elif lemmalist[idx].lower() in liwcdictnodiacritics:
            #Check if the lemma without diacritics is in LIWC
            liwcvalue = liwcdictnodiacritics[lemmalist[idx].lower()]
        if liwcvalue != None:
            for value in liwcvalue:
                newvalue = 'liwc' + str(value)
                if newvalue not in liwctotals:
                    liwctotals[newvalue] = 1
                else:
                    liwctotals[newvalue] += 1

    liwcvalues = {k: float(v) / float(len(wordlist)) for k, v in liwctotals.items()}
    return liwcvalues, liwctotals

def DutchFlemCategory(document):
    if 'Date' in document:
        return 'Date'
    elif 'LongTerm' in document:
        return 'LongTerm'

def SaveFeatures(): #This function saves after every 1000 iterations, which is nice if the database is huge,
    # so that you can stop/continue whenever you want, and to prevent too much damage after a crash
    currentpath = os.getcwd()
    #If the final files already exist, you don't need to do this whole process again
    #if os.path.isfile(currentpath + '/Corpus/train/FeaturesList/Categories.p'):
        #return 'Features already saved'

    #If there is no Featurefiles path yet in the current path, create one
    if not os.path.exists(currentpath + '/FeaturesList/'):
        os.makedirs(currentpath + '/FeaturesList/')
        print('FeaturesList path created')

    #Now let's get to work with the WordClustersLists we have
    wordngramslist = []
    wordngramstotallist = []
    liwcfeatureslist = []
    liwctotalfeatureslist = []
    categorieslist = []
    documentlist = ['DateTexts.p', 'LongTermTexts.p']
    #Go with one wordclusters file at the time, starting with the Casual (if there is no file yet)
    if not os.path.isfile(currentpath + '/FeaturesList/Categories.p'):
        #Load the LIWC-list
        liwcdict, liwcdictnodiacritics = LIWCdict(currentpath)

        for documentpath in documentlist:
            # Load the wordclusters
            with open(currentpath + '/Corpus/' + documentpath, 'rb') as f:
                wordclusterslist = pickle.load(f)

            print(documentpath + ' loaded')

            for idx, line in enumerate(wordclusterslist):
                #Extract the WordNgrams and append them to the list
                wordngramstotals, wordngrams = WordNGrams(line, 'y')
                wordngramslist.append(wordngrams)
                wordngramstotallist.append(wordngramstotals)
                # Extract the LIWC features and append them to the list
                liwcfeats, liwctotals = LIWCFeatures(line, liwcdict, liwcdictnodiacritics)
                liwcfeatureslist.append(liwcfeats)
                liwctotalfeatureslist.append(liwctotals)
                #Same idea for the categories
                categorieslist.append(DutchFlemCategory(documentpath))

                if ((idx % 10000 == 0) and (idx != 0)) or (idx == len(wordclusterslist) - 1):
                    print(str(idx+1) + '/' + str(len(wordclusterslist)))

        #Save the full feature files
        print('Saving WordNGrams features')
        with open(currentpath + '/FeaturesList/WordNGramsFeatures.p', 'wb') as f:
            pickle.dump(wordngramslist, f)

        with open(currentpath + '/FeaturesList/WordNGramsFeaturesTotals.p', 'wb') as f:
            pickle.dump(wordngramstotallist, f)

        print('Saving LIWC features')
        with open(currentpath + '/FeaturesList/LIWCFeatures.p', 'wb') as f:
            pickle.dump(liwcfeatureslist, f)

        with open(currentpath + '/FeaturesList/LIWCFeaturesTotals.p', 'wb') as f:
            pickle.dump(liwctotalfeatureslist, f)

        print('Saving categories')
        with open(currentpath + '/FeaturesList/Categories.p', 'wb') as f:
            pickle.dump(categorieslist, f)

#SaveFeatures()