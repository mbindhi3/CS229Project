from __future__ import division
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
mpl.use('Agg')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import os, itertools, subprocess
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import csv

def load_data():
    df = pd.read_csv('Final_filtered_data.csv')
    df.sample(frac=1)
    raw_gen=pd.concat([df.iloc[:,1:5],df.iloc[:,6:14]],axis=1,join='inner')
    raw_MRI=df.iloc[:,14:136]
    raw_data=pd.concat([raw_gen,raw_MRI],axis=1,join='inner')
    return raw_data

def preprocess_data(raw_data):
    # Dropping missing values
    raw_data_cleaned=raw_data.dropna()
    raw_data_cleaned=raw_data_cleaned[(raw_data_cleaned.iloc[:,14:136]!=' ').all(1)]

    # Set some features as categorical
    xcat_p = raw_data_cleaned[['PTGENDER','PTMARRY']]
    raw_data_cleaned.drop(['DX_bl','PTGENDER','PTMARRY'], axis=1, inplace=True)
    #PTGENDER: 0:Female; 1: Male -- #PTMARRY: 0:Divorced; 1: Married; 2: Never Married 4:Widowed

    y_p = raw_data_cleaned[['DX']]
    raw_data_cleaned.drop(['DX'], axis=1, inplace=True)
    #DX: 0: Dementia, 1:MCI to Dementia; 2: MCI; 3: NL

    le = preprocessing.LabelEncoder()
    xcat=xcat_p.apply(le.fit_transform)
    x=pd.concat([xcat,raw_data_cleaned],axis=1,join='inner')

    # Set 'DX' (Demented or Not) as categorical
    y=y_p.apply(le.fit_transform)
    return x,y


def split_data(x,y):
    train_split=0.8 # fraction of the data used in the training set
    m=x.shape[0] # number of data points

    x_train=x.iloc[0:int(m*train_split),:]
    y_train=y.iloc[0:int(m*train_split),:]
    x_test=x.iloc[int(m*train_split)+1:m-1,:]
    y_test=y.iloc[int(m*train_split)+1:m-1,:]
    return x_train, y_train, x_test, y_test

def run_PCA_LDA(X,y,xtest,ytest,components):
    y=np.ravel(y)
    target_names = ['Dementia','MCI','NL','MCI to Dementia']

    pca = PCA(n_components=components)
    pca1 =  pca.fit(X)
    X_r = pca1.transform(X)
    Xtest_r = pca1.transform(xtest)

    lda = LinearDiscriminantAnalysis(n_components=components)
    lda1= lda.fit(X, y)
    X_r2 = lda1.transform(X)
    Xtest_r2 = lda1.transform(xtest)

    # Percentage of variance explained for each component
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Tadpole dataset')

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Tadpole dataset')

    plt.show()

    x_pca=pd.DataFrame(X_r)
    x_lda=pd.DataFrame(X_r2)
    xtest_pca=pd.DataFrame(Xtest_r)
    xtest_lda=pd.DataFrame(Xtest_r2)
    y=pd.DataFrame(y)
    ytest = pd.DataFrame(ytest)
    return x_pca,x_lda,xtest_pca,xtest_lda

def feature_importances(x, clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    header_sorted=[]
    for f in range(len(list(x))):
        header_sorted.append(list(x)[indices[f]])
        print("%d. Feature: %s (%f)" % (f + 1, list(x)[indices[f]], importances[indices[f]]))

    # Plot the feature importances
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(x.shape[1]), header_sorted)
    plt.xlim([-1, x.shape[1]])
    plt.savefig('feature_importance_tadpole.png')
    plt.show()

def add_intercept(X_):
    #####################
    m = X_.shape[0]
    X = np.ones((m, 4))
    X[:, 1:4] = X_
    ###################
    return X


def dist(a, b):
    dist = 0
    ################
    dist = np.sum((a-b) * (a-b))
    ################
    return dist

def findNNOutput(X,y, Xtest, ytest):
    
    clf = MLPClassifier(hidden_layer_sizes=(5,10,10,), alpha=0.001,
                    solver='lbfgs', verbose=10, tol=1e-4, learning_rate ='invscaling',
                    learning_rate_init=1, max_iter =  1e07, activation = 'logistic',
                    random_state = 1)
    

    print(clf.fit(X, y))
    outputs =  clf.predict(X)
    print(outputs);
    y_pred = clf.predict(Xtest)
    print("Training set score: %f" % clf.score(X, y))
    print("Test set score: %f" % clf.score(Xtest, ytest))
    

    
    # Confusion Matrix
    cnf_matrix=confusion_matrix(ytest, y_pred)
    #DX: 0: Dementia, 1:MCI to Dementia; 2: MCI; 3: NL
    class_names=list(['Dementia','MCI','NL','MCI to Dementia'])
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('dementia.png')

def main():
    
    raw_data=load_data()
    x,y=preprocess_data(raw_data)
    x_train, y_train, x_test, y_test=split_data(x,y)
    x_pca,x_lda, xtest_pca, xtest_lda=run_PCA_LDA(x_train,y_train,x_test, y_test,components=10)
    findNNOutput(x_lda, y_train, xtest_lda, y_test)
    
    

if __name__ == '__main__':
    main()