import xlrd
import os
import regex as re

currentpath = os.getcwd()

def ConvertWorkbook(db):
    workbook = xlrd.open_workbook(d b)
    worksheets = workbook.sheet_names()[0]
    excellist = []
    # Open the excel file
    worksheet = workbook.sheet_by_name(worksheets)
    #Get all the rows
    for row in range(worksheet.nrows):
        #Excellist has the format [[relationship_type row 0, text row 0], [relationship_type row 1, text row 1], ...]
        #So first we are going to create the inner lists
        rowlist = []
        #Append the relationship type (first column)
        rowlist.append(worksheet.cell_value(row, 0))
        #Append the text (second column)
        rowlist.append(worksheet.cell_value(row, 1))
        #Add the rowinfo to the total list
        excellist.append(rowlist)

    return excellist

def WriteTxtFiles(excellist):
    #Keep track of the numbers for the title of the file
    datelist = []
    longtermlist = []

    for text in excellist:
        # Lots of these texts are cut off early, which makes the last word incomplete, if the last word is some text with ellipses like 'text...', delete the last word
        m = re.search(r"^(.*?)\s(\w+)[.]{3}$", text[1])
        if m:
            text[1] = m.group(1)
        #And also remove the ellipses at the end if they do not break up the last word
        text[1] = re.sub(r"[.]{3}$", '', text[1])
        #If the first column is a 0, the text is saved in the casual folder
        if text[0] == 0:
            datelist.append(text[1])
        #If the first column is a 1, the text is saved in the long term folder
        elif text[0] == 1:
            longtermlist.append(text[1])
    # Let's write the files now!
    datelist = '\n\n'.join(datelist)
    longtermlist = '\n\n'.join(longtermlist)
    with open(currentpath + '/DateTexts.txt', 'wb') as f:
        f.write(bytes(datelist, 'UTF-8'))
    with open(currentpath + '/LongTermTexts.txt', 'wb') as f:
        f.write(bytes(longtermlist, 'UTF-8'))

excellist = ConvertWorkbook(currentpath + "/Longterm_Date.xlsx")
WriteTxtFiles(excellist)

