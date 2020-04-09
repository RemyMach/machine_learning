import torch
import torch.nn.functional as F

#constuit un pytorch vecteur de 784 ligne et 10 colonne avec toujours ce calcule de gradient pour chaque valeur
weights = torch.randn(784, 10, requires_grad=True)
print(weights.shape)


#fonction de test pour calculer la précision que notre modèle a sur nos données de test
#notre fonction a besoin des poids et du test_loader qui va permettre de pouvoir itérer, boucler dans nos données de test
def test(weights, test_loader):
    test_size = len(test_loader.dataset)
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        #batch_idx : le numéro de notre tour
        #data.shape
        print(batch_idx, data.shape, target.shape)
        break

        data = data.view((-1, 28*28))
        #print(batch_idx, data.shape, target.shape)

        outputs = torch.matmul(data, weights)
        softmax = F.softmax(outputs, dim=1)
        pred = softmax.argmax(dim=1, keepdim=True)
        n_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += n_correct

    acc = correct / test_size
    print(" Accuracy on test set", acc)
    return

test(weights, test_loader)