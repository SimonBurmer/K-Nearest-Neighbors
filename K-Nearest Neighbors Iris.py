import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
from numpy import where
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

#implementing the k-nearest neighbors algorithm to predict the right iris species.

#The data set consists of 50 samples from each of three species of iris 
# (Iris setosa, Iris virginica and Iris versicolor). 
# Four features were measured from each sample: the length and 
# the width of the sepals and petals, in centimeters.

#load Data
iris = datasets.load_iris()
iris_data = iris.data  # iris_data is a npArray


#SHOW DATASET:

iris_taget = iris.target  # iris_taget is a npArray
#to store differten data types we have to convert the array into a list
iris_taget_list = iris_taget.tolist()

for i in range(len(iris_taget_list)):
    if iris_taget_list[i] == 0:
        iris_taget_list[i] = "Iris setosa"
    if iris_taget_list[i] == 1:
        iris_taget_list[i] = "Iris virginica"
    if iris_taget_list[i] == 2:
        iris_taget_list[i] = "Iris versicolor"

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.insert(4, "Iris", iris_taget_list)

print("--------------------------------------------")
print(df.head())



#PLOT DATA:

#Create color maps to plot colored points
cmap_dark = ListedColormap(['red', 'blue', 'orange', 'green'])

fig1, axs = plt.subplots(3, 2, num="Iris-Data", tight_layout= True)
axs[0, 0].scatter(iris_data[:, 0], iris_data[:, 1],c=iris_taget, cmap=cmap_dark, edgecolor='k', s=20)
axs[0, 0].set(xLabel="sepal length", yLabel="sepal width")
axs[0, 1].scatter(iris_data[:, 0], iris_data[:, 2],c=iris_taget, cmap=cmap_dark, edgecolor='k', s=20)
axs[0, 1].set(xLabel="sepal length", yLabel="petal length")
axs[1, 0].scatter(iris_data[:, 0], iris_data[:, 3],c=iris_taget, cmap=cmap_dark, edgecolor='k', s=20)
axs[1, 0].set(xLabel="sepal length", yLabel="pepal width")
axs[1, 1].scatter(iris_data[:, 1], iris_data[:, 2],c=iris_taget, cmap=cmap_dark, edgecolor='k', s=20)
axs[1, 1].set(xLabel="sepal width", yLabel="petal length")
axs[2, 0].scatter(iris_data[:, 1], iris_data[:, 3],c=iris_taget, cmap=cmap_dark, edgecolor='k', s=20)
axs[2, 0].set(xLabel="sepal width", yLabel="petal widht")
axs[2, 1].scatter(iris_data[:, 2], iris_data[:, 3],c=iris_taget, cmap=cmap_dark, edgecolor='k', s=20)
axs[2, 1].set(xLabel="pepal length", yLabel="petal widht")

plt.show()

#EXPLANATION SCATTER:
    #S = point size
    #edgecolor = border color of a point
    #c = A sequence of n numbers to be mapped to colors using cmap and norm.
    #cmap = A Colormap instance
    #Explanation:
        #c=[1,1,2,0,0,....1,2] cmap_dark= ['red', 'blue', 'orange']
        #the programm takes the first x, the first y and the first c value,
        #then is takes the right color corresponding to the c value from the cmap_dark
        # in this case, blue (because c=1). so the point is colored blue.



#MODEL TRAINING:

#split train/test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(iris_data, iris_taget, test_size=0.50)
#find out the best model params
best_acc = 0
for neighbors in range(3, 10):
    for weights in ['uniform', 'distance']:
        #model looks for the 3-10 closest datapoints
        model = KNeighborsClassifier(n_neighbors=neighbors, weights=weights)
        #train the model
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)

        if acc > best_acc:
            best_acc = acc
            best_model = model

model = best_model
#show best model params
print("--------------------------------------------")
print("best model params: \n" + str(model.get_params()))
print("aaccuracy: " + str(best_acc))



#PLOT PREDICITON AND ORIGINAL

#plot iris data
fig2, (axs1, axs2) = plt.subplots(1, 2, tight_layout=True, num = "Prediciton")
axs1.scatter(iris_data[:, 0], iris_data[:, 1],c=iris_taget, cmap=cmap_dark, edgecolor='k', s=20)
axs1.set(xLabel="sepal length", yLabel="sepal width")
axs1.set_title('data')

#plot predicted data 
predicton = model.predict(iris_data)
axs2.scatter(iris_data[:, 0], iris_data[:, 1],c=predicton, cmap=cmap_dark, edgecolor='k', s=20)
axs2.set(xLabel="sepal length", yLabel="petal length")
axs2.set_title('prediction')

plt.show()



