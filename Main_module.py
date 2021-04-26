from Get_Features import SaveFeatures
from Vectorize_features import SaveAll
from Run_classifier import RunAll
from Frequency_counts_and_graphs import Get_Ratio

#Save a list of dictionaries containing all the raw features
SaveFeatures()

#Vectorize these features
SaveAll()

#Run the classifier
RunAll()

#And collect the most important features, change the number of features by changing the number after n
Get_Ratio(n=100)