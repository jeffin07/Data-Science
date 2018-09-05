from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

#load data

digits=load_digits()


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target)

model=LogisticRegression()
model.fit(x_train,y_train)
print model.predict(x_test[0].reshape(1,-1))
#Build confussion matrix