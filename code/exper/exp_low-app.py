import os
import csv
import autograd.numpy as np
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import FixedRankEmbedded

from algorithms import ConjugateGradient, BetaTypes

from alg import ConjugateGradient1, BetaTypes1
from algsw import ConjugateGradient2, BetaTypes2
def create_cost(A):
    @pymanopt.function.Autograd
    def cost(u, s, vt):
        X = u @ np.diag(s) @ vt
        return np.linalg.norm(X - A) ** 2
    
    return cost


if __name__ == "__main__":
    experiment_name = 'low-app'
    n_exp = 15

    if not os.path.isdir('res1'):
        os.makedirs('res1')
    path = os.path.join('res1', experiment_name + '.csv')

    m, n, rank = 100, 80, 4
    
    for i in range(n_exp):
        matrix = rnd.randn(m, n)

        cost = create_cost(matrix)
        manifold = FixedRankEmbedded(m, n, rank)
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
