#Importaciones necesarias de librerias
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle


#Datasets 

iris = datasets.load_iris()

X = iris.data
y = iris.target

#split datasets

x_train, x_test, y_train, y_test = train_test_split(X, y)

lin_reg = LinearRegression()
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
svc_m = SVC()

# train the models

lin_regression = lin_reg.fit(x_train, y_train)
log_regression = log_reg.fit(x_train, y_train)
svc_model = svc_m.fit(x_train, y_train)


with open('Lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regression, li)

with open('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regression, lo)

with open('svc_model.pkl', 'wb') as sv:
    pickle.dump(svc_model, sv)

