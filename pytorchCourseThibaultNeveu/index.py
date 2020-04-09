import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = torch.tensor(3, dtype=torch.float, requires_grad=True)
y = torch.tensor(3, dtype=torch.float, requires_grad=True)

for i in range(100):
    # Compute the operation. We want to minimise the result. To do so we can compute
    # the gradient of each variable and apply the gradient descent formula.
    result = x ** 2 + y ** 2
    print("Result = >>", result)

    # Compute the gradient for each operations made before
    #result.backward calcule le gradient des variables de l'opération comme ceux-sont des variables
    # torch.tensor avec requires_grad à true
    #backward ne s'applique que si on a une fonction par exemple de type grad_fn sur le résultat
    #ainsi celui-ci remontera sur toutes les opérations pour calculer les gradient de toutes les variables torch.tensor
    result.backward()
    #on affiche le gradient de x donc la dérivée de x**2 ou x au carré est 2x donc 2*3 ainsi le gradient de x vaut 6
    print(x)
    # on affiche le gradient de y donc la dérivée de y**2 est 2y donc 2*3 ainsi le gradient de y vaut 6 aussi
    print(y)
    #result n'est qu'un résultat donc n'est pas une tensor varible mais ici un integer donc pas de gradient
    print(result.grad)

    #Apply gradient descent without tracking the gradient.
    #on ne track pas le gradient car dans cet exemple on ne cherche pas à le minimiser
    with torch.no_grad():
        x -= 0.1 * x.grad
        y -= 0.1 * y.grad

    #on reset les deux gradient car sinon chaque valeur va s'ajouter à chaque tour ainsi x vaudra 6 puis 12 puis 18...
    #donc on observe que on ajoute au gradient précédemment calculé la valeur du gradient de la variable donc 6
    #pour que cell-ci se reset à zéro on utilise .zero_()
    x.grad.zero_()
    y.grad.zero_()
