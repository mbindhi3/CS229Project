from __future__ import division
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFE
from sklearn.utils import shuffle

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
from sklearn.model_selection import ShuffleSplit
import csv

def load_data():
    df = pd.read_csv('TADPOLE_D1_D2.csv',low_memory=False)
    df = shuffle(df)
    raw_gen=df.loc[:,['AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY','APOE4','DX']]
    raw_cognitive_test=df.loc[:,['CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']]
    raw_MRI=df.loc[:,['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform','MidTemp']]
    raw_PET=df.loc[:,['FDG','AV45']]
    raw_CSF=df.loc[:,['ABETA_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17']]

    # Other Raw Data
    raw_other=df.loc[:,['CEREBELLUMGREYMATTER_UCBERKELEYAV45_10_17_16',
        'WHOLECEREBELLUM_UCBERKELEYAV45_10_17_16',
        'ERODED_SUBCORTICALWM_UCBERKELEYAV45_10_17_16',
        # 'COMPOSITE_REF_UCBERKELEYAV45_10_17_16', # NOT AVAILABLE
        'FRONTAL_UCBERKELEYAV45_10_17_16',
        'CINGULATE_UCBERKELEYAV45_10_17_16',
        'PARIETAL_UCBERKELEYAV45_10_17_16',
        'TEMPORAL_UCBERKELEYAV45_10_17_16',
        'SUMMARYSUVR_WHOLECEREBNORM_UCBERKELEYAV45_10_17_16',
        'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF_UCBERKELEYAV45_10_17_16',
        'SUMMARYSUVR_COMPOSITE_REFNORM_UCBERKELEYAV45_10_17_16',
        'SUMMARYSUVR_COMPOSITE_REFNORM_0.79CUTOFF_UCBERKELEYAV45_10_17_16',
        'BRAINSTEM_UCBERKELEYAV45_10_17_16',
        'BRAINSTEM_SIZE_UCBERKELEYAV45_10_17_16',
        'VENTRICLE_3RD_UCBERKELEYAV45_10_17_16',
        'VENTRICLE_3RD_SIZE_UCBERKELEYAV45_10_17_16',
        'VENTRICLE_4TH_UCBERKELEYAV45_10_17_16',
        'VENTRICLE_4TH_SIZE_UCBERKELEYAV45_10_17_16',
        'VENTRICLE_5TH_UCBERKELEYAV45_10_17_16',
        'VENTRICLE_5TH_SIZE_UCBERKELEYAV45_10_17_16',
        'CC_ANTERIOR_UCBERKELEYAV45_10_17_16',
        'CC_ANTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
        'CC_CENTRAL_UCBERKELEYAV45_10_17_16',
        'CC_CENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CC_MID_ANTERIOR_UCBERKELEYAV45_10_17_16',
        'CC_MID_ANTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
        'CC_MID_POSTERIOR_UCBERKELEYAV45_10_17_16',
        'CC_MID_POSTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
        'CC_POSTERIOR_UCBERKELEYAV45_10_17_16',
        'CC_POSTERIOR_SIZE_UCBERKELEYAV45_10_17_16',
        'CSF_UCBERKELEYAV45_10_17_16',
        'CSF_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_BANKSSTS_UCBERKELEYAV45_10_17_16',
        'CTX_LH_BANKSSTS_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_CAUDALANTERIORCINGULATE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_CAUDALANTERIORCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_CAUDALMIDDLEFRONTAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_CAUDALMIDDLEFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_CUNEUS_UCBERKELEYAV45_10_17_16',
        'CTX_LH_CUNEUS_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_ENTORHINAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_ENTORHINAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_FRONTALPOLE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_FRONTALPOLE_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_FUSIFORM_UCBERKELEYAV45_10_17_16',
        'CTX_LH_FUSIFORM_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_INFERIORPARIETAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_INFERIORPARIETAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_INFERIORTEMPORAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_INFERIORTEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_INSULA_UCBERKELEYAV45_10_17_16',
        'CTX_LH_INSULA_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_ISTHMUSCINGULATE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_ISTHMUSCINGULATE_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_LATERALOCCIPITAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_LATERALOCCIPITAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_LATERALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_LATERALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_LINGUAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_LINGUAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_MEDIALORBITOFRONTAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_MEDIALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_MIDDLETEMPORAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_MIDDLETEMPORAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_PARACENTRAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_PARACENTRAL_SIZE_UCBERKELEYAV45_10_17_16',
        'CTX_LH_PARAHIPPOCAMPAL_UCBERKELEYAV45_10_17_16',
        'CTX_LH_PARAHIPPOCAMPAL_SIZE_UCBERKELEYAV45_10_17_16',
        ]]

    raw_data=pd.concat([raw_gen,raw_cognitive_test,raw_MRI,raw_PET,raw_CSF,raw_other],axis=1,join='inner')
    return raw_data

def collapse_dx(raw_data):
  ret = pd.DataFrame.copy(raw_data)
  ret = ret[ret['DX'] != 'NL to MCI']
  ret = ret[ret['DX'] != 'MCI to NL']

  if 'DX' in ret:
    ret['DX'][ret['DX'] == 'MCI to Dementia'] = 'Dementia'
    ret['DX'][ret['DX'] == 'MCI'] = 'Dementia'
  return ret

def preprocess_data(raw_data):
    
    # Drop missing values
    raw_data_cleaned=raw_data.dropna(how='any')

    #raw_data_cleaned=raw_data_cleaned[(raw_data_cleaned!=' ').all(1)]

    # Convert 'DX' to 2 labels only: MCI is considered Dementia
    raw_data_cleaned=conv_binary_opp(raw_data_cleaned)

    # Set some features as categorical
    xcat_p = raw_data_cleaned[['PTGENDER','PTMARRY','APOE4']]
    raw_data_cleaned.drop(['PTGENDER','PTMARRY','APOE4'], axis=1, inplace=True)
    #PTGENDER: 0:Female; 1: Male -- #PTMARRY: 0:Divorced; 1: Married; 2: Never Married 4:Widowed

    y_p = raw_data_cleaned[['DX']]
    raw_data_cleaned.drop(['DX'], axis=1, inplace=True)
    #DX: 0: Dementia, 1:Normal

    le = preprocessing.LabelEncoder()
    xcat=xcat_p.apply(le.fit_transform)
    x=pd.concat([xcat,raw_data_cleaned],axis=1,join='inner')

    # Set 'DX' (Demented or Not) as categorical
    y=y_p.apply(le.fit_transform)
    comb=pd.concat([x,y],axis=1,join='inner')
    clean_comb=clean_data(comb)

    y = clean_comb[['DX']]
    clean_comb.drop(['DX'], axis=1, inplace=True)
    return clean_comb,y

def clean_data(raw_data):
    xnum= raw_data.apply(pd.to_numeric, errors='coerce')
    xnum = xnum.dropna()
    return xnum

def conv_binary(raw_data_cleaned):
    # Converting 'DX' to 2 labels only: MCI is considered Dementia
    raw_data_cleaned=raw_data_cleaned.replace('Dementia to MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to Dementia', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('NL to MCI', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to NL', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('NL to Dementia', 'Dementia')
    return raw_data_cleaned

def conv_binary_opp(raw_data_cleaned):
    # Converting 'DX' to 2 labels only: MCI is considered NL
    raw_data_cleaned=raw_data_cleaned.replace('Dementia to MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to Dementia', 'Dementia')
    raw_data_cleaned=raw_data_cleaned.replace('NL to MCI', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('MCI to NL', 'NL')
    raw_data_cleaned=raw_data_cleaned.replace('NL to Dementia', 'Dementia')
    return raw_data_cleaned


def split_data(x,y):
    train_split=0.7# fraction of the data used in the training set
    m=x.shape[0] # number of data points

    x_train=x.iloc[0:int(m*train_split),:]
    y_train=y.iloc[0:int(m*train_split),:]
    x_test=x.iloc[int(m*train_split)+1:m-1,:]
    y_test=y.iloc[int(m*train_split)+1:m-1,:]
    return x_train, y_train, x_test, y_test

def run_PCA_LDA(X,y,x_test, y_test, components):
    y=np.ravel(y)
    target_names = ['Dementia','Normal']

    pca = PCA(n_components=components)
    pca_fit = pca.fit(X);
    X_r = pca_fit.transform(X);
    xtest_r = pca_fit.transform(x_test)

    lda = LinearDiscriminantAnalysis(n_components=components)
    lda_fit= lda.fit(X, y);
    X_r2 = lda_fit.transform(X)
    xtest_r2 = lda_fit.transform(x_test)

    # Percentage of variance explained for each component
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2


    plt.show()

    x_pca=pd.DataFrame(X_r)
    x_pca_test = pd.DataFrame(xtest_r)
    x_lda=pd.DataFrame(X_r2)
    x_lda_test = pd.DataFrame(xtest_r2)

    return pca,lda,x_pca,x_lda, x_pca_test, x_lda_test

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
    
    clf = MLPClassifier(hidden_layer_sizes=(3, 2,), alpha=0.0001,
                    solver='lbfgs', verbose=10, tol=1e-4, learning_rate ='invscaling',
                    learning_rate_init=0.01, max_iter =  10000, activation = 'logistic',
                    random_state = 1)
    
    
    

    clf.fit(X, y)
    outputs =  clf.predict(X)
    #print(outputs);
    y_pred = clf.predict(Xtest)
    #print("Training set score: %f" % clf.score(X, y))
    #print("Test set score: %f" % clf.score(Xtest, ytest))
    

    
    """# Confusion Matrix
    cnf_matrix=confusion_matrix(y, outputs)
    #DX: 0: Dementia, 1:MCI to Dementia; 2: MCI; 3: NL
    class_names=list(['Dementia','Normal'])
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
    
    # Confusion Matrix
    cnf_matrix=confusion_matrix(ytest, y_pred)
    #DX: 0: Dementia, 1:MCI to Dementia; 2: MCI; 3: NL
    class_names=list(['Dementia','Normal'])
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')"""
    return clf.score(X, y), clf.score(Xtest, ytest)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.figure()
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
    plt.savefig('dementia_Tadpole.png')
    
def hold_out_CV(x, y):
    splits=50
    rs = ShuffleSplit(n_splits=splits, test_size=.3, random_state=0)
    sum_train =0;
    sum_test = 0;
    for train_index, test_index in rs.split(x):
        pca,lda,x_pca_train,x_lda_train, x_pca_test, x_lda_test=run_PCA_LDA(x.iloc[train_index],y.iloc[train_index],x.iloc[test_index], y.iloc[test_index], components=10)
        [training_score, testing_score] = findNNOutput(x_lda_train, y.iloc[train_index], x_lda_test, y.iloc[test_index])
        sum_train = sum_train + training_score;
        sum_test = sum_test + testing_score;
        
    print("Average Training Score : ", sum_train/splits)
    print("Average Training Score : ", sum_test/splits)
    

def main():
    
    raw_data=load_data()
    x,y=preprocess_data(raw_data)
    hold_out_CV(x,y)

    
    
    

    

if __name__ == '__main__':
    main()