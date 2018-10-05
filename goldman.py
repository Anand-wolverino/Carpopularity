

import pandas as pd
from sklearn import preprocessing, svm
import numpy as np
from sklearn.model_selection import train_test_split



c=1


#Read the csv file using pandas
cs=pd.read_csv("")#set path based on your requirements
cs=cs[['buying_price','maintainence_cost','number_of_doors','number_of_seats','luggage_boot_size','safety_rating','popularity']]
#store the features that needs to be used for training in x
x=np.array(cs.drop(['popularity'],1))
#store the class/variable to be predicted in y
y=np.array(cs['popularity'])

#split the data for training & validation 80%-training 20%-validation
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


#Fit the data to the SVM model
clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
z=clf.score(X_test, y_test)   
print('Accuracy:',z)


# readaing the test data 
names=['buying_price','maintainence_cost','number_of_doors','number_of_seats','luggage_boot_size','safety_rating']
cs2=pd.read_csv("",names=names)#set path 
cs2=cs2[['buying_price','maintainence_cost','number_of_doors','number_of_seats','luggage_boot_size','safety_rating']]
x_test=np.array(cs2)

#predicting the class 
pred=clf.predict(x_test)
print(pred)

#writing the class data to csv file
np.savetxt(" ",pred,fmt='%d')#set the path where you want to store
