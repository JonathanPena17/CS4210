#-------------------------------------------------------------------------
# AUTHOR: Jonathan Peña
# FILENAME: decision_tree_2.py
# SPECIFICATION: Complete the Python program (decision_tree_2.py) that will read the files
#contact_lens_training_1.csv, contact_lens_training_2.csv, and contact_lens_training_3.csv. Each of
#those training sets has a different number of instances. You will observe that now the trees are being
#created setting the parameter max_depth = 3, which it is used to define the maximum depth of the tree
#(pre-pruning strategy) in sklearn. Your goal is to train, test, and output the performance of the 3 models
#created by using each training set on the test set provided (contact_lens_test.csv). You must repeat
#this process 10 times (train and test by using a different training set), choosing the average accuracy
#as the final classification performance of each model.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['/Users/jonathanpena/Documents/CS4210/Assignment2/Question2/contact_lens_training_1.csv','/Users/jonathanpena/Documents/CS4210/Assignment2/Question2/contact_lens_training_2.csv', '/Users/jonathanpena/Documents/CS4210/Assignment2/Question2/contact_lens_training_3.csv']

def transform_features(features):
    feature_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, 'Yes': 1, 'No': 2, 'Myope': 4, 'Hypermetrope': 5, 'Reduced': 6, 'Normal': 7}
    return [feature_map.get(f, -1) for f in features]

def transform_class(label):
    return 1 if label == "Yes" else 2

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for row in dbTraining:
        X.append(transform_features(row[:-1]))

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for row in dbTraining:
        Y.append(transform_class(row[-1]))

    accumulated_accuracy = 0.0
    for i in range (10):
       correct_predictions = 0
       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = []
       with open('/Users/jonathanpena/Documents/CS4210/Assignment2/Question2/contact_lens_test.csv', 'r') as testfile:
        testreader = csv.reader(testfile)
        for j, testrow in enumerate(testreader):
            if j > 0:  # skipping the header
                dbTest.append(testrow)

       for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           test_features = transform_features(data[:-1])
           class_predicted = clf.predict([test_features])[0]

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
           true_label = transform_class(data[-1])
           if class_predicted == true_label:
            correct_predictions += 1

       #find the average of this model during the 10 runs (training and test set)
       #--> add your Python code here
       accuracy = correct_predictions / len(dbTest)
       accumulated_accuracy += accuracy

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    average_accuracy = accumulated_accuracy / 10
    print(f"Final average accuracy when training on {ds}: {average_accuracy:.2f}")




