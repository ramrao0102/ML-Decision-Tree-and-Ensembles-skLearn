# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 20:13:25 2022

@author: ramra
"""

# HW6 for CS5033

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import math
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

df = pd.read_csv('C:/Data Science and Analytics/CS 5033/Homeworks/Data_for_UCI_named.csv')

df.drop(columns=['p1'], inplace=True)
df.drop(columns=['stab'], inplace=True)

print(df.head())

# Exercise 1

for i in df.index:
    if df['stabf'][i] == 'stable':
        df['stabf'][i] = 1
    else:
        df['stabf'][i] = 0
        

print(df.head())

X = df.iloc[:, :-1]

Y = df.iloc[:,-1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2)

print(X_train.head(), len(X_train))

print(X_test.head(), len(X_test))

print(y_train.head(), len(y_train))

print(y_test.head(), len(y_test))

X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, random_state = 1, test_size = 0.25)

print(X_train1.head(), len(X_train1))

print(X_val1.head(), len(X_val1))

print(y_train1.head(), len(y_train1))

print(y_val1.head(), len(y_val1))

# Exercise 4

# Model 1
# Max_depth = 1, n_estimators = 20

ada_clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth =1), n_estimators = 20)

ada_clf1.fit(X_train1, y_train1)

y_val11 = y_val1.to_numpy()

# Model 1 on Validation Examples

probability1 = ada_clf1.predict_proba(X_val1)

cr_entropy = []

actual = []

for i in range(len(y_val1)):
    a = [0 for j in range(2)]
    a[y_val11[i]] = 1
    actual.append(a)

actual = np.array(actual)

probability11 = np.zeros((len(y_val1) ,2))

for i in range(len(probability1)):
   cross_entropy =0.0
   for j in range(len(probability1[i])):
       if probability1[i][j] ==0:
           probability11[i][j] = probability1[i][j] + 1e-15
       elif probability1[i][j] ==1:
           probability11[i][j] = probability1[i][j] - 1e-15 
       else:
           probability11[i][j] = probability1[i][j]
       cross_entropy += - actual[i][j]*math.log(probability11[i][j])
       
   cr_entropy.append(cross_entropy) 

sum =0.0
for i  in range(len(cr_entropy)):
    sum += cr_entropy[i]

mean = sum/len(y_val11)

print("Mean Cross Entropy on Validation Dataset, Model 1")
print(mean)

print("_______________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Validation Dataset, Model 1")

print(log_loss(y_val11, probability1))

print("_______________________________________________________________________")

# Model 2
# Max_depth = 1, n_estimators = 40

ada_clf2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth =1), n_estimators = 40)

ada_clf2.fit(X_train1, y_train1)

y_val11 = y_val1.to_numpy()


probability2 = ada_clf2.predict_proba(X_val1)

cr_entropy1 = []

actual = []

probability22 = np.zeros((len(y_val1) ,2))

for i in range(len(y_val1)):
    a = [0 for j in range(2)]
    a[y_val11[i]] = 1
    actual.append(a)

actual = np.array(actual)

for i in range(len(probability2)):
   cross_entropy =0.0
   for j in range(len(probability2[i])):
      if probability2[i][j] ==0:
          probability22[i][j] = probability2[i][j] + 1e-15
      elif probability2[i][j] ==1:
          probability22[i][j] = probability2[i][j] - 1e-15 
      else:
          probability22[i][j] = probability2[i][j]
      cross_entropy += - actual[i][j]*math.log(probability22[i][j])
       
   cr_entropy1.append(cross_entropy) 

sum =0.0
for i  in range(len(cr_entropy1)):
    sum += cr_entropy1[i]

mean = sum/len(y_val11)

print("Mean Cross Entropy on Validation Dataset, Model 2")
print(mean)

print("_______________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Validation Dataset, Model 2")

print(log_loss(y_val11, probability2))

print("_______________________________________________________________________")

# Model 3
# Max_depth = 1, n_estimators = 80

ada_clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth =1), n_estimators = 80)

ada_clf3.fit(X_train1, y_train1)

y_val11 = y_val1.to_numpy()

# Model 1 on Validation Examples

probability3 = ada_clf3.predict_proba(X_val1)

cr_entropy2 = []

actual = []

probability33 = np.zeros((len(y_val1) ,2))

for i in range(len(y_val1)):
    a = [0 for j in range(2)]
    a[y_val11[i]] = 1
    actual.append(a)

actual = np.array(actual)

for i in range(len(probability3)):
   cross_entropy =0.0
   for j in range(len(probability3[i])):
       if probability3[i][j] ==0:
           probability33[i][j] = probability3[i][j] + 1e-15
       elif probability3[i][j] ==1:
           probability33[i][j] = probability3[i][j] - 1e-15 
       else:
           probability33[i][j] = probability3[i][j]
       cross_entropy += - actual[i][j]*math.log(probability33[i][j])
           

       
   cr_entropy2.append(cross_entropy) 

sum =0.0
for i  in range(len(cr_entropy2)):
    sum += cr_entropy2[i]

mean = sum/len(y_val11)

print("Mean Cross Entropy on Validation Dataset, Model 3")
print(mean)

print("_______________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Validation Dataset, Model 3")

print(log_loss(y_val11, probability3))

print("_______________________________________________________________________")

# Model 1 outperforms Models #2 and #3 on Validation Datasets
# Now fitting to train+ validation dataset and testing on test dataset


ada_clf1.fit(X_train, y_train)

y_test1 = y_test.to_numpy()


probability4 = ada_clf2.predict_proba(X_test)

cr_entropy1 = []

actual = []

probability44 = np.zeros((len(y_test) ,2))

for i in range(len(y_test)):
    a = [0 for j in range(2)]
    a[y_test1[i]] = 1
    actual.append(a)

actual = np.array(actual)

for i in range(len(probability4)):
   cross_entropy =0.0
   for j in range(len(probability4[i])):
       if probability4[i][j] ==0:
           probability44[i][j] = probability4[i][j] + 1e-15
       elif probability4[i][j] ==1:
           probability44[i][j] = probability4[i][j] - 1e-15 
       else:
           probability44[i][j] = probability4[i][j]
       cross_entropy += - actual[i][j]*math.log(probability44[i][j])
       
   cr_entropy1.append(cross_entropy) 

sum =0.0
for i  in range(len(cr_entropy1)):
    sum += cr_entropy1[i]

mean = sum/len(y_test1)

print("Mean Cross Entropy on Test Dataset, Model 1")
print(mean)

print("_______________________________________________________________________")

print("Log_Loss from Sk Learn In Built Function on Test Dataset, Model 1")

print(log_loss(y_test1, probability4))

print("_______________________________________________________________________")

# Model 1 Prediction of Labels Based on Probability Thresholds, Validation Dataset

predictions1 = []
for i in range(0,1001,1):
    k = float(i/1000)
    label1 = []
    for j in range(len(probability1)):
        
        if probability1[j][1] > k:
            label1.append(1)
        else:
            label1.append(0)

    predictions1.append(label1)
    
  

# Model 2 Prediction of Labels Based on Probability Thresholds, Validation Dataset

predictions2 = []

for i in range(0,1001,1):
    k = float(i/1000)
    label2 = []
    for j in range(len(probability2)):
        
        if probability2[j][1] > k:
            label2.append(1)
        else:
            label2.append(0)

    predictions2.append(label2)
    
#print(len(predictions2)) 

#print(predictions2[0])

#print(predictions2[1])   
    

# Model 3 Prediction of Labels Based on Probability Thresholds, Validation Dataset

predictions3 = []

for i in range(0,1001,1):
    k = float(i/1000)
    label3 = []
    for j in range(len(probability3)):
        
        if probability3[j][1] > k:
            label3.append(1)
        else:
            label3.append(0)

    predictions3.append(label3)
    

# Model 1 on Train + Validation Dataset    
    
  
predictions4= []

for i in range(0,1001,1):
    k = float(i/1000)
    label4 = []
    for j in range(len(probability4)):
        
        if probability4[j][1] > k:
            label4.append(1)
        else:
            label4.append(0)

    predictions4.append(label4)
    
# Random Predictor

RX = []
RY = []

for i in range(0,1001, 1):
    k = float(i/1000)
            
    RX.append(k)
    RY.append(k)
    
# compute True Positive Rate and True Negative Rate 

# Model 1 Prediction of Labels Based on Probability Thresholds

TPR1 = []
FPR1 = []
Youden_Index1 = 0

for i in range(0,1001, 1):
    k = float(i/1000)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for j in range(len(predictions1[i])):
        if predictions1[i][j] ==0:
            if y_val11[j] == 0:
                tn += 1
            else:
                fn += 1
        
        if predictions1[i][j] == 1:
            if y_val11[j]  == 1:
                tp += 1
            else:
                fp += 1
        
    if (tp+fn) != 0:
        true_positive_rate1 = tp/(tp+fn)
    else:
        true_positive_rate1 =0
    
    
    
    if (tn +fp) !=0:
        false_positive_rate1 = fp/(tn+fp)
    else:
        false_positive_rate1 = 0
    
    if (true_positive_rate1 - false_positive_rate1) > Youden_Index1:
        Youden_Index1 = (true_positive_rate1 - false_positive_rate1)
        final_index = k
    
    
    TPR1.append(true_positive_rate1)
    FPR1.append(false_positive_rate1)

print("Probability Threshold with Highest Index", final_index)            
plt.plot(FPR1, TPR1)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["Ensemble Model 1 Validation Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for Model 1 Validation Dataset is:")
print(Youden_Index1)
        
# Model 2 Prediction of Labels Based on Probability Thresholds

TPR2 = []
FPR2 = []
Youden_Index2 = 0.0

for i in range(0,1001, 1):
    k = float(i/1000)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for j in range(len(predictions2[i])):
        if predictions2[i][j] ==0:
            if y_val11[j] == 0:
                tn += 1
            else:
                fn += 1
        
        if predictions2[i][j] == 1:
            if y_val11[j]  == 1:
                tp += 1
            else:
                fp += 1
        
    if (tp+fn) != 0:
        true_positive_rate2 = tp/(tp+fn)
    else:
        true_positive_rate2 =0
    
   
    
    if (tn +fp) !=0:
        false_positive_rate2 = fp/(tn+fp)
    else:
        false_positive_rate2 = 0
    
    if (true_positive_rate2 - false_positive_rate2) > Youden_Index2:
       Youden_Index2 = (true_positive_rate2 - false_positive_rate2)
       final_index = k
    
    TPR2.append(true_positive_rate2)
    FPR2.append(false_positive_rate2)

print("Probability Threshold with Highest Index", final_index)          
plt.plot(FPR2, TPR2)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["Ensemble Model 2 Validation Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for Model 2 Validation Dataset is:")
print(Youden_Index2)

#  Model 3 Prediction of Labels Based on Probability Thresholds

TPR3 = []
FPR3 = []
Youden_Index3 = 0.0

for i in range(0,1001, 1):
    k = float(i/1000)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for j in range(len(predictions3[i])):
        if predictions3[i][j] ==0:
            if y_val11[j] == 0:
                tn += 1
            else:
                fn += 1
        
        if predictions3[i][j] == 1:
            if y_val11[j]  == 1:
                tp += 1
            else:
                fp += 1
        
    if (tp+fn) != 0:
        true_positive_rate3 = tp/(tp+fn)
    else:
        true_positive_rate3 =0
    
   
    
    if (tn +fp) !=0:
        false_positive_rate3 = fp/(tn+fp)
    else:
        false_positive_rate3 = 0
    
    if (true_positive_rate3 - false_positive_rate3) > Youden_Index3:
      Youden_Index3 = (true_positive_rate3 - false_positive_rate3)
      final_index = k
    
    TPR3.append(true_positive_rate3)
    FPR3.append(false_positive_rate3)

print("Probability Threshold with Highest Index", final_index)             
plt.plot(FPR3, TPR3)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["Ensemble Model 3 Validation Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for Model 3 Validation Dataset is:")
print(Youden_Index3)  

# Model 1 Prediction of Labels Based on Probability Thresholds for Train + Validation Dataset

TPR4 = []
FPR4 = []
Youden_Index4 = 0.0

for i in range(0,1001, 1):
    k = float(i/1000)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for j in range(len(predictions4[i])):
        if predictions4[i][j] ==0:
            if y_test1[j] == 0:
                tn += 1
            else:
                fn += 1
        
        if predictions4[i][j] == 1:
            if y_test1[j]  == 1:
                tp += 1
            else:
                fp += 1
        
    if (tp+fn) != 0:
        true_positive_rate4 = tp/(tp+fn)
    else:
        true_positive_rate4 =0
    
   
    
    if (tn +fp) !=0:
        false_positive_rate4 = fp/(tn+fp)
    else:
        false_positive_rate4 = 0
    
    if (true_positive_rate4 - false_positive_rate4) > Youden_Index4:
      Youden_Index4 = (true_positive_rate4 - false_positive_rate4)
      final_index = k
    
    TPR4.append(true_positive_rate4)
    FPR4.append(false_positive_rate4)

print("Probability Threshold with Highest Index", final_index)           
plt.plot(FPR4, TPR4)
plt.plot(RX, RY, c='0.85')
plt.xlabel("False Positive Rate.", size = 8,)
plt.ylabel("True Positive Rate", size = 8)
plt.legend(["Ensemble Model 1 Test Dataset"], loc ="lower right", prop = {'size': 8})
plt.show()   

print("Highest Youden_Index for Model 1 Test Dataset is:")
print(Youden_Index4)  

# Area Under Curve Determinations

# for Model 1

AUC1 = 0.0

for i in range(len(FPR1)):

    if i == 0:
        
        prev_coordinate = FPR1[i]
        AUC1 =0.0

    if i >0:
    
        AUC1 += (1/2)* (TPR1[i] +TPR1[i])*(-FPR1[i] + prev_coordinate)   


    prev_coordinate = FPR1[i]

print("AUC from Fuction Built from Raw FPR and TPR Data, Model 1 Validation Dataset")
    
print(AUC1)

print("_____________________________________________________________________________")

print("AUC from In Built Function, Model 1 Validation Dataset")

auc = roc_auc_score(y_val11, probability1[:,1])

print(auc)

print("_____________________________________________________________________________")

# for Model 2

AUC2 = 0.0

for i in range(len(FPR2)):

    if i == 0:
        
        prev_coordinate = FPR2[i]
        AUC2 =0.0

    if i >0:
    
        AUC2 += (1/2)* (TPR2[i] +TPR2[i])*(-FPR2[i] + prev_coordinate)   


    prev_coordinate = FPR2[i]

print("AUC from Fuction Built from Raw FPR and TPR Data, Model 2 Validation Dataset")
    
print(AUC2)

print("_____________________________________________________________________________")

print("AUC from In Built Function, Model 2 Validation Dataset")

auc = roc_auc_score(y_val11, probability2[:,1])

print(auc)

print("_____________________________________________________________________________")


# for Model 3 Prediction on Train Dataset

AUC3 = 0.0

for i in range(len(FPR3)):

    if i == 0:
        
        prev_coordinate = FPR3[i]
        AUC3 =0.0

    if i >0:
    
        AUC3 += (1/2)* (TPR3[i] +TPR3[i])*(-FPR3[i] + prev_coordinate)   


    prev_coordinate = FPR3[i]

print("AUC from Fuction Built from Raw FPR and TPR Data, Model 3 Validation Dataset")

print(AUC3)

print("_____________________________________________________________________________")

print("AUC from In Built Function, Model 3 Validation Dataset")

auc = roc_auc_score(y_val11, probability3[:,1])

print(auc)

print("_____________________________________________________________________________")

# for Model 1 Prediction on Train + Validation Dataset

AUC4 = 0.0

for i in range(len(FPR4)):

    if i == 0:
        
        prev_coordinate = FPR4[i]
        AUC4 =0.0

    if i >0:
    
        AUC4 += (1/2)* (TPR4[i] +TPR4[i])*(-FPR4[i] + prev_coordinate)   

    prev_coordinate = FPR4[i]

print("AUC from Fuction Built from Raw FPR and TPR Data, Model 1 Test Dataset")

print(AUC4)

print("_____________________________________________________________________________")

print("AUC from In Built Function, Model 1 Test Dataset")

auc = roc_auc_score(y_test1, probability4[:,1])

print(auc)


print("_____________________________________________________________________________")
