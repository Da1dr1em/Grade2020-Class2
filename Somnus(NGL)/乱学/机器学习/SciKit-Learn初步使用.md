[API 参考-scikit-learn中文社区](https://scikit-learn.org.cn/lists/3.html)
[[API 参考-scikit-learn中文社区]]  ---> 本地的
```python
import numpy as np  
import sklearn.datasets  
from sklearn import datasets  
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  
Iris = sklearn.datasets.load_iris()  ##使用datasets中的Iris数据做示范
Iris_x = Iris.data  ##数据集
Iris_y = Iris.target  ##数据集对应结果
  
X_train, X_test, Y_train, Y_test = train_test_split(Iris_x,Iris_y,test_size=0.7)  
##这个模块打乱并分开用作测试的数据和用作训练的数据  
KNC = KNeighborsClassifier()  ##定义一下函数
KNC.fit(X_train,Y_train)  ##训练
print("目标编号: " , Y_test)  
print("预测结果: " , KNC.predict(X_test)) ##预测并输出
```