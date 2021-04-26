import os
import csv
from ftfy import fix_encoding

currentpath = os.getcwd()

liwcdict = {}

with open(currentpath + '/LIWC-Features/LIWC-Compleet_WithCategories_WithAddedWordsAttractiveness.txt', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')

    for idx, row in enumerate(reader):
        print(row)
        if idx > 0:
            newrow = list(filter(None, row))
            liwcdict.update({newrow[0]: newrow[1:]})

print(liwcdict)