from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

# Le dataset principal qui contient toutes les images
print (mnist.data.shape)

# Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
print (mnist.target.shape)

#knn = neighbors.KNeighborsClassifier(n_neighbors=3)
#knn.fit(xtrain, ytrain)
#error = 1 - knn.score(xtest, ytest)
#print('Erreur: %f' % error)
errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
plt.plot(range(2,15), errors, 'o-')
plt.show()