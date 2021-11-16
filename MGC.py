# -*- coding: utf-8 -*-
"""
@author: cmbbd
"""
import pandas as pd
import numpy as np 

import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import sklearn.decomposition as sd
import tensorflow as tf

from tensorflow import keras

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

#Chargement des features (30 Sec ou 3 sec)

#df1 = pd.read_csv('Data/Feature_3sec.csv')
#df1 = pd.read_csv('Data/FEATURE_3sec_delta2.csv')
df1 = pd.read_csv('Data/features_3_sec.csv')






#Preparation des données

le = preprocessing.LabelEncoder()
le.fit(df1.label)
df1['categorical_label'] = le.transform(df1.label) #Label de 0 à 9

df2 = df1.drop(['filename','length'],axis = 1)

#df2 = df1


#Mettre tous les features à la même échelle
scaler = MinMaxScaler()
df2[df2.drop(['label','categorical_label'],axis = 1).columns] = scaler.fit_transform(df2[df2.drop(['label','categorical_label'],axis = 1).columns])

#X = df2.drop(['label','categorical_label','Unnamed: 0'],axis = 1)
X = df2.drop(['label','categorical_label'],axis = 1)
y = df1['categorical_label']








#Division de la base de donnée en train, test et val

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5)
x_test,x_val,y_test1,y_val = train_test_split(X_test,y_test,test_size=0.333,random_state=5)













#Construction d'un réseau de neurones
def build_nn():
    model = keras.Sequential([
        keras.layers.Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation='relu'),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


#Transformation du modèle tensorflow en classifier sklearn

model1 = tf.keras.wrappers.scikit_learn.KerasClassifier(
                            build_nn,
                            epochs=200,
                            verbose=False)
model1._estimator_type = "classifier"

#Evaluation du réseau de neurones
model = build_nn()
model.fit(X_train,y_train,verbose=0,epochs=100)
yp = model.predict_classes(x_test)

y_pred = []


for element in yp:
    y_pred.append(element)
from sklearn.metrics import confusion_matrix , classification_report
Report = classification_report(y_test1,y_pred)
cm = confusion_matrix(y_test1,y_pred)







#Deuxième modèles K nearest neighbors     
      
from sklearn.neighbors import KNeighborsClassifier

SCORE = []
for k in range(1,18):
        # Division de la base de données
        (valData, test1Data,valLabels, test1Labels)  = train_test_split(x_test,y_test1 , test_size=0.4, random_state = 42)
        clf = KNeighborsClassifier(n_neighbors= k)
        clf.fit(X_train, y_train)
        predict = clf.predict(valData)
        SCORE.append(clf.score(valData,valLabels))
clf = KNeighborsClassifier(n_neighbors= np.argmax(SCORE)+2)
clf.fit(X_train,  y_train)

#Evaluation KNN
predictions2 = clf.predict(x_test)
Report1 = classification_report(y_test1,predictions2)
cm1 = confusion_matrix(y_test1,predictions2)





#Creation d'un modèle (Voting classifier), pondérant les résultats du KNN et NN

MODEL_pred = model.predict_proba(x_val)
CLF_pred = clf.predict_proba(x_val)
ALPHA = [i/1000 for i in range(1000)]
LOSS = [sk.metrics.log_loss(y_val,(alpha*CLF_pred+(1-alpha)*MODEL_pred)) for alpha in ALPHA]
alpha = ALPHA[np.argmin(LOSS)]

#Evaluation du modèle pondéré
MODEL_pred = model.predict_proba(x_test)
CLF_pred = clf.predict_proba(x_test)
Report2 = classification_report(y_test1,np.argmax(alpha*CLF_pred+(1-alpha)*MODEL_pred,axis=1))
cm2 = confusion_matrix(y_test1,np.argmax(alpha*CLF_pred+(1-alpha)*MODEL_pred,axis=1))

#Creation Voting Classifier Sklearn

vot = VotingClassifier([('1',clf),('2',model1)],voting = 'soft')
vot.fit(X_train, y_train)

#Evaluation du modèle
predict = vot.predict(x_test)
Report4 = classification_report(y_test1,predict)
cm4 = confusion_matrix(y_test1,predict)

