import tensorflow
import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read in file
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]
predict = "G3"

X=np.array(data.drop(predict,1))
y=np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.05)

# best_score = 0
# for i in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.05)
#
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train,y_train)
#
#     acc = linear.score(x_test, y_test)
#     print(acc)
#     if(acc>best_score):
#         best_score=acc
#         with open("student-model.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("student-model.pickle","rb")

linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(round(predictions[x],0), x_test[x], y_test[x])

style.use("ggplot")
pyplot.scatter(data["G1"],data["G2"],data[predict])
pyplot.xlabel("G1")
pyplot.ylabel("Final Grade")
pyplot.show()
