# On importe les librairies dont on aura besoin pour ce tp
#%matplotlib inline
#A executer sur Jupyter NoteBook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D

################### Première phase: exploration/nettoyage des données ###################
#chargement des données
raw_data = pd.read_csv('house_data_more_precise.csv')

#résumé des données brutes, avec ça on voit que i a pas le même nombre de ligne donc on va trier
#print(raw_data.describe())

#quelques valeurs manquante donc on supprime des lignes
data_na = raw_data.dropna()

#on enlève les outliers qui sont dans notre dataset sur les grandes propriétés
data = data_na[data_na["price"] < 8000]

#on réindexe
data = data.reset_index(drop = True)

print(data)

#on affiche les données nettoyés
print(data.plot.scatter('price','surface', c='arrondissement', colormap='viridis'))

################### Affichage de la varable prédite(price) en fonction de l'arrondissement ###################
#affichage du graphique
ax1 = sns.violinplot(x='arrondissement', y='price', data=data, hue= 'arrondissement')
#affiche sur le graphique des petite graduation sur l'axe x et y mais aussi à l'interieur du graphique  (des grandes lignes verticale)
ax1.minorticks_on()
#affiche les lignes verticale qui sépare nos schémas
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
#ordonne l'axe mineur si il existe pour le schéma
ax1.grid(which='minor', axis='x', linewidth=1)


################### On croise le prix avec la surface pour avoir une vision plus claire ###################
#affcihage du graphe 3D
fig = plt.figure().gca(projection='3d')

# Pour faciliter la visualisation, on va changer la valeur de l'arrondissement (10) en 5
#on récupère la valeur de tous les indices
tmp_arr = data['arrondissement'][:]
#et on change celle-ci
tmp_arr[tmp_arr == 10] = 5

#set tous les points du schéma 3D fig.scatte(axe x, axe z, axe y), c défini en fonction de quoi les couleurs change pour différencier les points
fig.scatter(tmp_arr, data['surface'], data['price'], c=tmp_arr, cmap="viridis")
#on affiche le graphe 3D
plt.show()


################### régression spécifique pour séparer jeu de données d'entrainement/jeu de données de test ###################

#dans les x on met surface et arrondissement et dans les y on met le price et on prend 30% des données pour le test et 70% pour le train
xtrain, xtest, ytrain, ytest = train_test_split(data[["surface", "arrondissement"]], data[["price"]], test_size=0.3)

print(xtrain)

################### On crée ensuite la baseline puis on calcule la R2( la somme quadratique des résidus), comme valeur d'évaluation de notre regression ###################

#calcul de la somme quadratique
lr = LinearRegression()
lr_baseline = lr.fit(xtrain[["surface"]], ytrain)
baseline_pred = lr_baseline.predict(xtest[["surface"]])

#on affiche le graphique
#affichage des points qui représente les données
plt.plot(xtest[["surface"]], ytest, 'bo', markersize = 5)
#affichage de la droite qui représente la moyenne de la surface en fonction du prix
plt.plot(xtest[["surface"]], baseline_pred, color="skyblue", linewidth = 2)

#on peut réécrire rapidement le calcul du R2 score:
#c'est avec ces 2 fonctions que l'on obtient le R2 score donc le taux de précision de notre algorithme
def sumsq(x,y):
    return sum((x - y)**2)

def r2score(pred, target):
    return 1 - sumsq(pred, target) / sumsq(target, np.mean(target))

score_bl = r2score(baseline_pred[:,0], ytest['price'])

#print(score_bl)

lrs = []
#on passe une fois sur chaque arrondissement avec cette boucle for
for i in np.unique(xtrain["arrondissement"]):

    # On génère un jeu de données par arrondissement
    tr_arr = xtrain['arrondissement']==i
    te_arr = xtest['arrondissement']==i
    #dans xtrain_arr on a toutes les surface avec les arrondissement groupby arrondissement
    #dans ytrain_arr on a tout les prices groupby arrondissement
    #tr_arr représente la clé de la ligne
    xtrain_arr = xtrain[tr_arr]
    ytrain_arr = ytrain[tr_arr]

    xtest_arr = xtest[te_arr]
    ytest_arr = ytest[te_arr]

    lr = LinearRegression()
    lr.fit(xtrain_arr[["surface"]], ytrain_arr)
    lrs.append(lr)

################### On effectue la prediction finale sur le jeu de donnée test avec le nouveau modèle, qui combine les différents modèles par arrondissement ###################
final_pred = []

for idx,val in xtest.iterrows():
    final_pred.append(lrs[int(val["arrondissement"]-1)].predict([[val["surface"]]])[0][0])

r2score(final_pred, ytest["price"])
print(r2score(final_pred, ytest["price"]))

################## On peut afficher la prediction finale #######################
#surface axe x, prix axe y.
plt.plot(xtest[["surface"]], ytest, 'bo', markersize = 5)
plt.plot(xtest[["surface"]], lrs[0].predict(xtest[["surface"]]), color="#00FFFF", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[1].predict(xtest[["surface"]]), color="#0000FF", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[2].predict(xtest[["surface"]]), color="#00FF00", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[3].predict(xtest[["surface"]]), color="#FF0000", linewidth = 2)
plt.plot(xtest[["surface"]], lrs[4].predict(xtest[["surface"]]), color="#FFFF00", linewidth = 2)

sys.exit(0)