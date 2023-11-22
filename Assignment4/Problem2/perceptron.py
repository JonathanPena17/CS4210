#-------------------------------------------------------------------------
# AUTHOR: Jonathan Peña
# FILENAME: perceptron.py
# SPECIFICATION: Complete the Python program (perceptron.py) that will read the file optdigits.tra to build a
#   Single Layer Perceptron and a Multi-Layer Perceptron classifiers. You will compare their performances
#   and test which combination of two hyperparameters (learning rate and shuffle) leads you to the best
#   prediction performance for each classifier. To test the accuracy of those distinct models, you will use
#   the file optdigits.tes. You should update and print the accuracy of each classifier, together with the
#   hyperparameters when it is getting higher.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('Assignment4/Problem2/optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('Assignment4/Problem2/optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

accuracy_perceptron = 0
params_perceptron = ()
accuracy_mlp = 0
params_mlp = ()

for i in n: #iterates over n

    for j in r: #iterates over r

        #iterates over both algorithms
        #-->add your Pyhton code here

        for k in ['Perceptron', 'MLP']: #iterates over the algorithms

            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
            #                          shuffle = shuffle the training data, max_iter=1000
            
            if k == 'Perceptron':
                clf = Perceptron(eta0=i, shuffle=j, max_iter=1000)
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=i, hidden_layer_sizes=(64,), shuffle=j, max_iter=1000)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            
            correct_predictions = 0
            for x_testSample, y_testSample in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                if prediction == y_testSample:
                    correct_predictions += 1
            accuracy = correct_predictions / len(y_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            
            if k == 'Perceptron' and accuracy > accuracy_perceptron:
                accuracy_perceptron = accuracy
                params_perceptron = (i, j)
                print(f"Highest Perceptron accuracy so far: {accuracy_perceptron:.6f}, Parameters: learning rate={i}, shuffle={j}")


            if k == 'MLP' and accuracy > accuracy_mlp:
                accuracy_mlp = accuracy
                params_mlp = (i, j)
                print(f"Highest MLP accuracy so far: {accuracy_mlp:.6f}, Parameters: learning rate={i}, shuffle={j}")