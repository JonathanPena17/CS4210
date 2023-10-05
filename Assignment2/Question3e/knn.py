#-------------------------------------------------------------------------
# AUTHOR: Jonathan Peña
# FILENAME: knn.py
# SPECIFICATION: Complete the Python program (knn.py) that will read the file binary_points.csv 
# and output the LOO-CV error rate for 1NN (same answer of part a (error rate for part a = 0.4)) 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('/Users/jonathanpena/Documents/CS4210/Assignment2/Question3e/binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

num_errors = 0
#loop your data to allow each instance to be your test set
for index, instance in enumerate(db):
    X = []
    Y = []
    for i, other_instance in enumerate(db):
        if i != index:
            #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
            # float to avoid warning messages
            #--> add your Python code here
            # X =
            X.append([float(other_instance[0]), float(other_instance[1])])
            #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
            #  feature value to float to avoid warning messages
            #--> add your Python code here
            # Y =
            if other_instance[2] == '+':
                Y.append(1)
            else:
                Y.append(2)

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [[float(instance[0]), float(instance[1])]]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here

    class_predicted = clf.predict(testSample)[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if (class_predicted == 1 and instance[2] != '+') or (class_predicted == 2 and instance[2] != '-'):
        num_errors += 1

#print the error rate
#--> add your Python code here
error_rate = num_errors / len(db)
print(f"Error rate: {error_rate}")





