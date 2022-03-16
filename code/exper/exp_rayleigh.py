import os
import csv
from sklearn.datasets import make_spd_matrix
import autograd.numpy as np
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Sphere

from algorithms import ConjugateGradient, BetaTypes
from alg import ConjugateGradient1, BetaTypes1
from algsw import ConjugateGradient2, BetaTypes2
def create_cost(A):
    @pymanopt.function.Autograd
    def cost(x):
        return np.inner(x, A @ x)

    return cost


if __name__ == "__main__":
    experiment_name = 'rayleigh'
    n_exp = 15

    if not os.path.isdir('res1'):
        os.makedirs('res1')
    path = os.path.join('res1', experiment_name + '.csv')

    n = 100
    
    for i in range(n_exp):
        matrix = make_spd_matrix(n)

        cost = create_cost(matrix)
        manifold = Sphere(n)
        problem = pymanopt.Problem(manifold, cost=cost, egrad=None)
        
        res_list = []

        for beta_type in BetaTypes:
            solver = ConjugateGradient(beta_type=beta_type, maxiter=10000)
            res1 = solver.solve(problem)
            res_list.append(res1[1])
            res_list.append(res1[2])
        solver1 = ConjugateGradient1(BetaTypes1.Hybrid1, maxiter=10000)
        res1 = solver1.solve(problem)
        res_list.append(res1[1])
        res_list.append(res1[2])
        solver2 = ConjugateGradient1(BetaTypes1.Hybrid2, maxiter=10000)
        res1 = solver2.solve(problem)
        res_list.append(res1[1])
        res_list.append(res1[2])
        #res_list.append(res[3])
        solver3 = ConjugateGradient1(BetaTypes1.Hybrid3, maxiter=10000)
        res1 = solver3.solve(problem)
        res_list.append(res1[1])
        res_list.append(res1[2])

        with open(path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(res_list)
