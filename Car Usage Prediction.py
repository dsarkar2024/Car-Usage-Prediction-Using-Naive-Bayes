#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:41:04 2023

"""

import itertools
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10f}'.format

from sklearn import preprocessing, metrics, naive_bayes


# Define a function to visualize the percent of a particular target category by a nominal predictor
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table for:\n", countTable)
   print( )

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

data = pandas.read_excel('E:\\Intro_to_Machine_Learning\\Assignment_3\\claim_history.xlsx')

# CAR_USE -> CAR_TYPE, EDUCATION, OCCUPATION
subData = data[['CAR_TYPE', 'EDUCATION', 'OCCUPATION', 'CAR_USE']].dropna()

catCarUSe = subData['CAR_USE'].unique()
catCarType = subData['CAR_TYPE'].unique()
catEducation = subData['EDUCATION'].unique()
catOccupation = subData['OCCUPATION'].unique()

print('Unique Values of Car Use: \n', catCarUSe)
print('Unique Values of Car Type: \n', catCarType)
print('Unique Values of Education: \n', catEducation)
print('Unique Values of Occupation: \n', catOccupation)

#RowWithColumn(rowVar = subData['CAR_USE'], columnVar = subData['CAR_TYPE'], show = 'ROW')
#RowWithColumn(rowVar = subData['CAR_USE'], columnVar = subData['EDUCATION'], show = 'ROW')
#RowWithColumn(rowVar = subData['CAR_USE'], columnVar = subData['OCCUPATION'], show = 'ROW')

train_data = data[['CAR_USE', 'CAR_TYPE', 'OCCUPATION', 'EDUCATION']].dropna().reset_index(drop = True)

yTrain = train_data['CAR_USE'].astype('category')

CAR_TYPE = train_data['CAR_TYPE'].astype('category')
OCCUPATION = train_data['OCCUPATION'].astype('category')
EDUCATION = train_data['EDUCATION'].astype('category')

# Train a Categorical Naive Bayes model

feature = ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']

featureCategory = [CAR_TYPE.cat.categories, OCCUPATION.cat.categories, EDUCATION.cat.categories]

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xTrain = featureEnc.fit_transform(train_data[feature])

_objNB = naive_bayes.CategoricalNB(alpha = 0.01)
thisModel = _objNB.fit(xTrain, yTrain)

# Print the counts and the row probabilities (after adjusted with alpha)


#question A
print("\n PART A \n")
print('Target Class Count')
print(thisModel.class_count_)

print('Target Class Probability')
print('Commericial, Private')
print(numpy.exp(thisModel.class_log_prior_))



RowWithColumn(rowVar = train_data['CAR_USE'], columnVar = train_data['CAR_TYPE'], show = 'ROW')
RowWithColumn(rowVar = train_data['CAR_USE'], columnVar = train_data['OCCUPATION'], show = 'ROW')
RowWithColumn(rowVar = train_data['CAR_USE'], columnVar = train_data['EDUCATION'], show = 'ROW')
'''
for i in range(len(feature)):
   print('Predictor: ', feature[i])
   print('Empirical Counts of Features')
   print(thisModel.category_count_[i])

   print('Empirical Probability of Features given a class, P(x_i|y)')
   print(numpy.exp(thisModel.feature_log_prob_[i]))
   print('\n')
'''
# C- Calculate the predicted probabilites on the first fictitious persons

xPerson_1 = pandas.DataFrame({'CAR_TYPE': ['SUV'],
                            'OCCUPATION': ['Skilled Worker'],
                            'EDUCATION': ['Doctors']})

xTest_1 = featureEnc.transform(xPerson_1)
yTest_predProb_1 = pandas.DataFrame(_objNB.predict_proba(xTest_1), columns = 'P_' + yTrain.cat.categories)

yTest_score_1 = pandas.concat([xPerson_1, yTest_predProb_1], axis = 1)
print('\nPredicted Probability for person works in a Skilled Worker occupation, has an education level of Doctors, and owns an SUV\n')
print(yTest_score_1)




#D- Calculate the predicted probabilites on the second fictitious persons
xPerson_2 = pandas.DataFrame({'CAR_TYPE': ['Sports Car'],
                            'OCCUPATION': ['Management'],
                            'EDUCATION': ['Below High School']})

xTest_2 = featureEnc.transform(xPerson_2)
yTest_predProb_2 = pandas.DataFrame(_objNB.predict_proba(xTest_2), columns = 'P_' + yTrain.cat.categories)

yTest_score_2 = pandas.concat([xPerson_2, yTest_predProb_2], axis = 1)
print('\n Predicted Probability for person works in a Management occupation, has an education level of Below High School, and owns an Sports Car\n')
print(yTest_score_2)

# Calculate the predicted probabilities on the training data


# Show the histogram of P_Private

yTrain_predProb = pandas.DataFrame(_objNB.predict_proba(xTrain), columns = 'P_' + yTrain.cat.categories)
predProbPrivate = yTrain_predProb['P_Private']
plt.hist(predProbPrivate, bins=[i/20 for i in range(21)], edgecolor='k', alpha=0.7)
plt.xlabel("Predicted Probability of CAR_USE = Private")
plt.ylabel("Proportion of Observations")
plt.title("Histogram of Predicted Probabilities")
plt.show()

# Calculate misclassification rate

y_Train_predCat = numpy.where(predProbPrivate >= 0.5, 'Private', 'Commercial')
#y_Train_MCE = 1.0 - metrics.accuracy_score(yTrain, y_Train_predCat)
#print(y_Train_MCE)


#F
print("\n PART F \n")
predicted_labels = thisModel.predict(xTrain)

misclassification_rate = 1 - metrics.accuracy_score(yTrain, predicted_labels)

print("Misclassification Rate of the Naïve Bayes model:", misclassification_rate)