
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.utils.data import evaluate_print
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
import warnings

warnings.filterwarnings("ignore")
from sklearn.externals import joblib
import os
import sys


# In[2]:


inlier_train = pd.read_csv('inlier.csv',header=None)
outlier_train =  pd.read_csv('outlier.csv',header=None)
#test =  pd.read_csv('test.csv',header=None)


# In[3]:


y_inlier=np.zeros( (inlier_train.shape[0],1), dtype=np.int16 )
y_outlier=np.ones( (outlier_train.shape[0],1), dtype=np.int16 )


# In[4]:


X= pd.concat([inlier_train,outlier_train], axis=0)


# In[5]:


y=np.append(y_inlier,y_outlier)


# In[6]:


# apply standard scaler to output from resnet50
ss = StandardScaler()
ss.fit(X)
x_train = ss.transform(X)
#test_x130 = ss.transform(test)#####小写der


# In[7]:


outlier_per=outlier_train.shape[0]/(outlier_train.shape[0]+inlier_train.shape[0])



sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))



X_train, X_test, y_train, y_test = train_test_split( x_train, y, test_size=0.3, random_state=42)



outliers_fraction = outlier_per  # percentage of outliers
clf_name = 'IForest'
clf = KNN(contamination=outliers_fraction)
clf.fit(X_train)
# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)


# In[12]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_test_pred)


# In[13]:


random_state = np.random.RandomState(42)
# Define nine outlier detection tools to be compared
classifiers = {
                # 'Angle-based Outlier Detector (ABOD)':
                #    ABOD(contamination=outliers_fraction),
               'Fast Angle-based Outlier Detector (FastABOD)':
                   ABOD(contamination=outliers_fraction),
               'Cluster-based Local Outlier Factor (CBLOF)':
                   CBLOF(contamination=outliers_fraction,
                         check_estimator=False, random_state=random_state),
               'Feature Bagging':
                   FeatureBagging(LOF(n_neighbors=35),
                                  contamination=outliers_fraction,
                                  check_estimator=False,
                                  random_state=random_state),
               'Histogram-base Outlier Detection (HBOS)': HBOS(
                   contamination=outliers_fraction),
               'Isolation Forest': IForest(contamination=outliers_fraction,
                                           random_state=random_state),
               'K Nearest Neighbors (KNN)': KNN(
                   contamination=outliers_fraction),
               'Average KNN': KNN(method='mean',
                                  contamination=outliers_fraction),
               'Median KNN': KNN(method='median',
                                 contamination=outliers_fraction),
               'Local Outlier Factor (LOF)':
                   LOF(n_neighbors=35, contamination=outliers_fraction),
               # 'Local Correlation Integral (LOCI)':
               #     LOCI(contamination=outliers_fraction),
              
               'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction,
                                              random_state=random_state),
               'Principal Component Analysis (PCA)': PCA(
                   contamination=outliers_fraction, random_state=random_state),
                'Auto encoder': AutoEncoder(epochs=30, contamination=outliers_fraction, random_state=random_state)
               # 'Stochastic Outlier Selection (SOS)': SOS(
               #     contamination=outliers_fraction),
               }





# Show all detectors
for i, clf in enumerate(classifiers.keys()):
    print('Model', i + 1, clf)

# Fit the models with the generated data and
# compare model performances

models={}
for i, (clf_name, clf) in enumerate(classifiers.items()):
    print()
    print(i + 1, 'fitting', clf_name)
    # fit the data and tag outliers
    clf.fit(X_train)
    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores
    print(confusion_matrix(y_test,y_test_pred))
    models[clf_name]=clf
   # joblib.dump(clf, 'model_%s.pkl' % (clf_name))
    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)


# In[23]:


full_test=pd.read_csv('test_full.csv',header=None)
submission=pd.DataFrame()
submission['ID']=full_test.iloc[:,-1]


# In[24]:


X_testfull=full_test.drop(columns=2048)
#X


# In[26]:


X_testfull=ss.transform(X_testfull)


# In[28]:


for i, (clf_name, clf) in enumerate(classifiers.items()):
    print()
    print(i + 1, 'predicting', clf_name)
    model= joblib.load('model_%s.pkl' % (clf_name))
    y_test_pred = model.predict(X_testfull) 
    y_test_scores = model.decision_function(X_testfull)
    submission[clf_name]=y_test_pred
submission.to_csv('submission.csv',index=False)


submission.to_csv('submission.csv',index=False)

