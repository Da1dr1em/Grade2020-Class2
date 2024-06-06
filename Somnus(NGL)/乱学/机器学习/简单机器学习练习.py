import numpy as np
import sklearn.datasets
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
Iris = sklearn.datasets.load_iris()
Iris_x = Iris.data
Iris_y = Iris.target

X_train, X_test, Y_train, Y_test = train_test_split(Iris_x,Iris_y,test_size=0.7)

KNC = KNeighborsClassifier()
KNC.fit(X_train,Y_train)
print("目标编号: " , Y_test)
print("预测结果: " , KNC.predict(X_test))