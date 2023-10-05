#-------------------------------------------------------------------------
# AUTHOR: Jonathan Peña
# FILENAME: naive_bayes.py
# SPECIFICATION: Complete the Python program (naïve_bayes.py) that will read the file
# weather_training.csv (training set) and output the classification of each test instance from the file
# weather_test (test set) if the classification confidence is >= 0.75. 
# Sample of output:
# Day Outlook Temperature Humidity Wind PlayTennis Confidence
# D15 Sunny Hot High Weak No 0.86
# D16 Sunny Mild High Weak Yes 0.78
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
with open('/Users/jonathanpena/Documents/CS4210/Assignment2/Question5b/weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    training_data = [row for row in reader]

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temp_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Weak': 1, 'Strong': 2}
play_map = {'Yes': 1, 'No': 2}

X = []
Y = []

for row in training_data[1:]:  # skip header
    X.append([outlook_map[row[1]], temp_map[row[2]], humidity_map[row[3]], wind_map[row[4]]])
    Y.append(play_map[row[5]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here

with open('/Users/jonathanpena/Documents/CS4210/Assignment2/Question5b/weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    test_data = [row for row in reader]

#printing the header os the solution
#--> add your Python code here

print("Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence")

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

for row in test_data[1:]:  # skip header
    sample = [[outlook_map[row[1]], temp_map[row[2]], humidity_map[row[3]], wind_map[row[4]]]]
    prediction_proba = clf.predict_proba(sample)[0]
    
    predicted_class = clf.predict(sample)[0]
    
    confidence = max(prediction_proba)
    
    if confidence >= 0.75:
        print(row[0], row[1], row[2], row[3], row[4], 'Yes' if predicted_class == 1 else 'No', round(confidence, 2))


