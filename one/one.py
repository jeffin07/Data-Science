from sklearn import svm
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import numpy as np
x=[[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39]]
x_test=[[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

y=['male', 'male', 'female', 'female', 'male', 'male', 'female' ]
y_test=['female',
'female', 'male', 'male']

clf=svm.SVC()
clf.fit(x,y)
y_pred=clf.predict(x_test)
score=accuracy_score(y_test,y_pred)
print score



clf3=SGDClassifier(loss="hinge", penalty="l1")
clf3.fit(x,y)
y_pred2=clf3.predict(x_test)
score1=accuracy_score(y_test,y_pred2)
print score1

clf4=neighbors.KNeighborsClassifier(4,weights="distance")
clf4.fit(x,y)
y_pred4=clf4.predict(x_test)
score2=accuracy_score(y_test,y_pred4)
print score2


i=np.argmax([score,score1,score2])
a={0:"SVM",1:"SGD",2:"KNeighbors"}

print "The high accuracy classifier for this example is : {}".format(a[i])