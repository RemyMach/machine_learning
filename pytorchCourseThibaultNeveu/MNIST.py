from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt

# Transform each image into tensor and normalized with mean and std
#pytorch transforme ces images qui sont de base des numpy array en tensor avec transforms.ToTensor() qui va convertir ces numpy array
#comme si c'était une boucle il en récupère un il le convertit et ainsi de suite jusquà qu'il n'y en ai plus
#le deuxième paramètre qui est tensor.Normalize normalise les données afin de converger bien plus rapidement, celui-ci prend
#deux paramètre la moyenne et l'écart-type
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# Define the batch size used each time we go through the dataset
batch_size = 32

# Set the training loader
#ce que notre ligne va faire c'est qu'elle va mettre les data dans le dossier ../data, donnnée d'entrainement, on les télécharge si on ne les a pas déjà,
#on définit la taille du lot qui est définit juste au dessus, le shuffle lui indique que à chaque fois qu'on ira dans notre dataset on prendra des données
#aléatoirement, dernier point le transform indique que l'on va modifier le dataset automatiquement avec la ligne transform du dessus
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
# Set the testing loader
#pour les données de test on fait exactement la même chose sauf que quand on le charge on met qu'on ne veut pas les données d'entrainement
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True)

#constuit un pytorch vecteur de 784 ligne et 10 colonne avec toujours ce calcule de gradient pour chaque valeur
#donc on note bien que les poids ce sont les variables d'entré.
weights = torch.randn(784, 10, requires_grad=True)
#print(weights.shape)


#fonction de test pour calculer la précision que notre modèle a sur nos données de test
#notre fonction a besoin des poids et du test_loader qui va permettre de pouvoir itérer, boucler dans nos données de test
def test(weights, test_loader):
    test_size = len(test_loader.dataset)
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        #batch_idx : le numéro de notre tour
        #data.shape: 32 valeur car on a défini test_loader avec un lot de 32 par 1
        # avec 28 par 28 qui sont les pixels de nos images
        #print(batch_idx, data.shape, target.shape)

        #on modifie la shape de data afin de ne plus avoir les images en 28*28
        #mais de les avoir en un vecteur de 784.
        #le -1 correspond au faite que l'on laisse torch automatiquement déduire quelle sera la shape
        #mais on lui impose le fait que chaque image doit maintenant faire 784 et plus 28 par 28
        #le neurone le plus stimulé, qui a reconnu le plus de pattern dans l'image sera le plus élevé
        data = data.view((-1, 28*28))
        #print(batch_idx, data.shape, target.shape)

        #data représente tous nos vecteur d'images * les poids qui leur sont associé
        #l'opération torch.matmul va nous permettre d'avoir les vecteurs d'activation linéaire  pour chacun des neurones
        #donc on fait ce torch.matmul de nos data * les poids qui sont associé
        #on obtient 32 valeurs et pour chacune de ces valeurs on a nos 10 valeurs d'activation qui représente nos classes
        #outputs[0] quand à lui va réprésenté les 10 valeurs linéaire du premier lot donc de la première classe
        outputs = torch.matmul(data, weights)
        #print(outputs.shape, outputs[0])

        #donc maintenant avec softmax on va essayer de ramener cette valeur linéaire en une probabilité
        #on l'applique à la dimension 1 car on veut l'appliquer à chacune des 10 valeures linéaire qui est à l'indice 1
        #de notre ouputs. Cela va nous ramener des valeurs entre 0 et 1
        #softmax[0] représentera donc les proba du premier lot
        softmax = F.softmax(outputs, dim=1)
        #print(softmax[0])
        #argmax permet de regarder sur nos 10 valeurs celle qui est la plus haute et quelle est la classe
        #qui a détecté le plus de pattern, toujours la dimension 1 pour les 10 valeurs que l'on a et on garde la même
        #dimension qu'on avait avant
        pred = softmax.argmax(dim=1, keepdim=True)
        #nous permet de print avec pred[0] la valeur qui a la plus grande probabilité
        #print(pred[0], pred.shape)
        # pred.eq test l'égalité entre notre target et notre prédiction
        #n_correct = pred.eq(target.view_as(pred))
        #maintenant on fait la somme de tout ça pour obtenir le nombre de bonne prédictions
        #n_correct = pred.eq(target.view_as(pred)).sum()
        #.item nous permet de resortir uniquement la valeur donc de ne plus avoir un n_correct = tensor(2) par exemple
        #mais juste égale à 2
        #peut-être un doute là-dessus, ce n_correct esst étrange car il resssort la valeur de l'item qu'on a et ça me semble étrange
        n_correct = pred.eq(target.view_as(pred)).sum().item()
        #print(n_correct)

        #on ajoute au compteur correct pour chaque tour le nombre de bonne prédictions que l'on a
        correct += n_correct

    #à la fin de la boucle pour calculer la précision de notre modèle on prend le nombre de bonne réponse divisé par la taille
    #de notre test_size donc la taille des données dont on a testé si la prediction était bonne ou non.
    #print(correct)
    acc = correct / test_size
    print(" Accuracy on test set", acc)
    return

test(weights, test_loader)

#on séléctionne le neurone qui a besoin d'être maximisé, dont la probabilité a besoin d'être maximisé,
#on va prendre le négative nll de ce neurone qui a besoin d'être maximisé, et en minimisant le nll on maximise la probabilité.
it = 0
for batch_idx, (data, targets) in enumerate(train_loader):
    # Be sure to start the loop with zeros grad
    #on oublie pas de réinitialisé le gradient pour qu'il soit recalculé depuis zéro
    #si grad déjà défini alors il on le remet à zéro.
    if weights.grad is not None:
        weights.grad.zero_()
    #on reforme nos données pour que notre photos soit une suite de 784 pixels et plus 28*28
    data = data.view((-1, 28 * 28))
    #print("batch_idx: {}, data.shape: {}, target.shape: {}".format(batch_idx, data.shape, targets.shape))
    #On calcul l'output de notre modèle donc nos 10 neurones de sortie
    outputs = torch.matmul(data, weights)
    # resultat  : torch.Size([32, 10])
    #print("outputs.shape: {}".format(outputs.shape))

    #ici on calcul le log du softmax car on a besoin de calculer l'erreur du modèle donc on aura l'erreur de cette probabilité
    #on rappelle que le softmax lui donne la probabilité pour toutes les classes
    #le F vient de torch.nn et qui donne accès au log_softmax
    #on applique donc ça sur les 10 valeurs qui représente nos 10 neurones de sorties qui sont les 10 neurones
    #qui gère les classe permetant de dire ok c'est un 0 un 1, un 2 ...
    #on calcule le log car celui-ci a de meilleur probabilité de convergence
    log_softmax = F.log_softmax(outputs, dim=1)
    softmax = F.softmax(outputs, dim=1)
    #resultat de : torch.Size([32, 10])
    #donc ici on ne fait rien à part afficher le format de ce qu'on a
    #print("softmax: {}".format(softmax[0]))
    #print("Log softmax: {}".format(log_softmax[0]))



    # print((-log_softmax[0][targets[0]] + -log_softmax[1][targets[1]] )  / 2 )
    #print(-log_softmax[0][targets[0]], targets[0])

    #le F.nll_loss indique l'erreur moyenne, ce qu'on veut donc minimiser
    #nll veut dire negativeloglikelyhood qui est la  probabilité qu'un neurone prédise telle classe
    #le faire comme ça permet de le faire sur tout le lot ainsi l'erreur va pouvoir être minimisé sur l'antierreté du lot
    loss = F.nll_loss(log_softmax, targets)
    #print("\rLoss shape: {}".format(loss), end="")


    # Compute the gradients for each variables
    #fait en sorte de calculer les gradients
    # ça va nous permettre donc d'avoir leur dérivé et donc la direction dans laquelle il faut modifier ces variables
    loss.backward()

    #maintenant  qu'on a calculé les gradient de nos poids on les modifie à l'aide du learning rate qui est 0.1
    #si 0.1 * weigths.grad est négatif alors ça ajoutera à weights donc les condition de la descente de gradient sont
    #bien respecté.
    with torch.no_grad():
        weights -= 0.1 * weights.grad

    it += 1
    #touts les 100 tours on appelle notre fonction test pour voir ou on en est par rapport à nos données test
    if it % 100 == 0:
        test(weights, test_loader)

    #on arrête notre boucle à 5000 tours
    if it > 5000:
        break

#on prend un lot dans nos données de test
batch_idx, (data, target) = next(enumerate(test_loader))

#on modifie le batch avec le data.view, son format
data = data.view((-1, 28*28))

#on calcule donc l'ouput donc la valeur linéaire à la sortie de notre lot
outputs = torch.matmul(data, weights)
#on calcule le softmax pour obtenir les probabilités en sortie
softmax = F.softmax(outputs, dim=1)
#on obtient la prédiction pour chacune des donnée en utilisant le argmax, donc on obtient avec pred[0] la milleur
pred = softmax.argmax(dim=1, keepdim=True)

#ici on affiche la première image de notre batch, notre lot
plt.imshow(data[0].view(28, 28), cmap="gray")
#et pour cette première image on affiche la prédiction effectué
plt.title("Predicted class {}".format(pred[0]))
#et enfin on montre l'image
plt.show()