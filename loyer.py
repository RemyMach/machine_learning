# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# On charge le dataset
house_data = pd.read_csv('house.csv')
house_data = house_data[house_data['loyer'] < 1000]
# On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].values]).T
y = np.matrix(house_data['loyer']).T
""" regr = linear_model.LinearRegression()
regr.fit(X, y)
regr.predict([[35]]) """

# On effectue le calcul exact du paramètre theta
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta.item(0) + theta.item(1) * 20)

# On affiche le nuage de points dont on dispose
#plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
#plt.show()

plt.xlabel('Surface')
plt.ylabel('Loyer')

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)

# On affiche la droite entre 0 et 250
plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')

plt.show()
