from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Classifiers():
    def __init__(self,data):
        ''' 
        TODO: Write code to convert the given pandas dataframe into training and testing data 
        # all the data should be nxd arrays where n is the number of samples and d is the dimension of the data
        # all the labels should be nx1 vectors with binary labels in each entry 
        '''

        X = data[['A','B']]
        y = data['label']

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
    
        self.training_data = np.asarray(X_train)
        self.training_labels = np.asarray(y_train)
        self.testing_data = np.asarray(X_test)
        self.testing_labels = np.asarray(y_test)
        self.outputs = []

        plt.figure(figsize=(10,7)) # Specify size of the chart
        plt.scatter('A', 'B', data=data[y==0], marker = 'o', c = 'blue')
        plt.scatter('A', 'B', data=data[y==1], marker = 'x', c = 'green')
        plt.title("A vs B for All Labels")
        plt.xlabel("A")
        plt.ylabel("B")
        plt.legend(('0', '1'), loc='center left', title='Class Label', bbox_to_anchor=(1, 0.5))
        #plt.savefig(f'Dataset.png')
        plt.show()
    
    def test_clf(self, clf, classifier_name):
        # TODO: Fit the classifier and extract the best score, training score and parameters

        if(classifier_name == 'K-Nearest Neighbors'):
            parameters = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'leaf_size': [5, 10, 15, 20, 25, 30]}
        elif(classifier_name == 'Logistic Regression'):
            parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
        elif(classifier_name == 'Decision Trees'):
            parameters = {'max_depth': [x for x in range(1,51)], 'min_samples_split': [x for x in range(2,11)]}
        elif(classifier_name == 'Random Forest'):
            parameters = {'max_depth': [x for x in range(1,6)], 'min_samples_split': [x for x in range(2,11)]}
        elif(classifier_name == 'Adaboost'):
            parameters = {'n_estimators': [x*10 for x in range(1,8)]}
            #print(parameters)

        grid_clf = GridSearchCV(estimator=clf, param_grid = parameters, cv=5)
        grid_clf.fit(self.training_data,self.training_labels)

        clf = grid_clf.best_estimator_
        #print(clf)
        clf.fit(self.training_data,self.training_labels)

        training_acc = "%0.3f" % (grid_clf.best_score_,)
        testing_acc = "%0.3f" % (clf.score(self.testing_data,self.testing_labels),)
        
        self.outputs.append(f'{classifier_name}, {training_acc}, {testing_acc}')

        # Use the following line to plot the results
        
        self.plot(self.testing_data, clf.predict(self.testing_data),model=clf,classifier_name=classifier_name)

    def classifyNearestNeighbors(self):
        # TODO: Write code to run a Nearest Neighbors classifier
        KNN = KNeighborsClassifier()
        self.test_clf(KNN,'K-Nearest Neighbors')
        
    def classifyLogisticRegression(self):
        # TODO: Write code to run a Logistic Regression classifier
        LR = LogisticRegression()
        self.test_clf(LR,'Logistic Regression')
    
    def classifyDecisionTree(self):
        # TODO: Write code to run a Logistic Regression classifier
        DT = DecisionTreeClassifier()
        self.test_clf(DT,'Decision Trees')

    def classifyRandomForest(self):
        # TODO: Write code to run a Random Forest classifier
        RF = RandomForestClassifier()
        self.test_clf(RF,'Random Forest')

    def classifyAdaBoost(self):
        # TODO: Write code to run a AdaBoost classifier
        ADA = AdaBoostClassifier()
        self.test_clf(ADA,'Adaboost')

    def plot(self, X, Y, model,classifier_name = ''):
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5

        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.

        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)

        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)

        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap = plt.cm.coolwarm, zorder = 1)
        plt.scatter(X1, X2, s = 38, c = Y)

        plt.margins(0.0)
        # uncomment the following line to save images

        #plt.savefig(f'{classifier_name}.png')
        plt.show()

    
if __name__ == "__main__":
    df = pd.read_csv('input.csv')
    models = Classifiers(df)
    print('Classifying with NN...')
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()

    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)