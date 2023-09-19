#-------------------------------------------------------------------------
# AUTHOR: Jonathan Peña
# FILENAME: decision_tree.py
# SPECIFICATION: Complete the given python program (decision_tree.py) that will read the file contact_lens.csv and output a decision tree.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('/Users/jonathanpena/Documents/CS4210/Assignment1/contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
# X =

for row in db:
    age = 0
    if row[0] == 'Young':
        age = 1
    elif row[0] == 'Prepresbyopic':
        age = 2
    elif row[0] == 'Presbyopic':
        age = 3

    spectacle = 1 if row[1] == 'Myope' else 2
    astigmatism = 1 if row[2] == 'No' else 2
    tear = 1 if row[3] == 'Reduced' else 2

    X.append([age, spectacle, astigmatism, tear])

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
# Y =
for row in db:
    Y.append(1 if row[4] == 'No' else 2)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()