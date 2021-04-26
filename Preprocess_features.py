from bs4 import BeautifulSoup
import pickle
import os
import sys

def wordclusterdict(wordcluster):
    worddict = {}
    t = wordcluster.find('t').text
    worddict.update({'t': t})
    posclass = wordcluster.find('pos')['class']
    worddict.update({'posclass': posclass})
    poshead = wordcluster.find('pos')['head']
    worddict.update({'poshead': poshead})
    lemma = wordcluster.find('lemma')['class']
    worddict.update({'lemma': lemma})
    return worddict

def extractwordclusters(soup):
    #Make a list of lists containing the folia-info per word
    wordclusterlist = []
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        paragraphclusters = []
        wordclusters = paragraph.find_all('w')
        for wordcluster in wordclusters:
            newwordcluster = wordclusterdict(wordcluster)
            paragraphclusters.append(newwordcluster)
        wordclusterlist.append(paragraphclusters)

    return wordclusterlist

currentpath = os.getcwd()
with open(currentpath + '/DateTexts.xml', 'rb') as f:
    soup = BeautifulSoup(f, "lxml")
wordclusters = extractwordclusters(soup)

print(len(wordclusters))
with open(currentpath + '/DateTexts.p', 'wb') as f:
    pickle.dump(wordclusters, f)
print('Wordclusters saved!')

with open(currentpath + '/LongTermTexts.xml', 'rb') as f:
    soup = BeautifulSoup(f, "lxml")
wordclusters = extractwordclusters(soup)

print(len(wordclusters))
with open(currentpath + '/LongTermTexts.p', 'wb') as f:
    pickle.dump(wordclusters, f)
print('Wordclusters saved!')