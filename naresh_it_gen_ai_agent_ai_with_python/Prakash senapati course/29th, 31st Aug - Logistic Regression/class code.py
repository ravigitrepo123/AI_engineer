import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\NIT\1. NIT_Batches\1. MORNING BATCH\N_Batch -- 7.30AM _ Dec26\3. Aug\29th, 31st - Logistic Regression\2.LOGISTIC REGRESSION CODE\logit classification.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20,
                                                    random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() 
classifier.fit(X_train, y_train) 

classifier.get_params()  

y_pred = classifier.predict(X_test)  


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac) 

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr) 

bias = classifier.score(X_train, y_train)
print(bias)

var = classifier.score(X_test, y_test)
print(var) 
















