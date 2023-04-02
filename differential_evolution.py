from random import randint
import numpy as np

def differential_evolution_minimize(f, lb, ub, tol=1e-6, n_iter_max=1000, n_individuals=0, crossover_proba=0.9, differential_weight=0.8):
    ''' Minimizes the function f using the differential evolution global optimization algorithm.
        f                   function to minimize
        lb                  Nx1 vector of lower bounds of the N-dimensional domain
        ub                  Nx1 vector of upper bounds of the N-dimensional domain
        tol                 tolerance on average function value difference between successive iterations
        n_iter_max          maximum number of iterations to perform
        n_individuals       number of individuals to use in the population. By default, =10*n_dims
        crossover_proba     probability of performing a cross-over for a given coordinate of an individual. Must be in [0 ; 1]
        differential_weight coefficient in front of the differential term (b_i - c_i). Usually in [0 ; 2] 
    '''
    res = {'x': None, 'fx': None, 'n_iter': n_iter_max, 'converged': False}

    # Pre checks
    if len(lb) != len(ub):
        print(f'Error : lb and ub must have the same sizes ! lb : {len(lb)}, ub : {len(ub)}')
        return res

    lb = np.asarray(lb)
    ub = np.asarray(ub)

    n_dims = len(lb)

    if n_individuals < 4:
        n_individuals = 10*n_dims

    # initialize the population
    population = []
    population_values = []
    for i in range(n_individuals):
        x = np.random.rand(n_dims)*(ub - lb) + lb
        population.append(x)
        population_values.append(f(x))
    
    for iter in range(n_iter_max):
        # for each individual
        for ix, x in enumerate(population):
            # pick 3 other individuals a, b, and c
            ia = ix
            ib = ix
            ic = ix
            while ia == ix:
                ia = int(np.random.rand() * n_individuals)
            while ib == ix or ib == ia:
                ib = int(np.random.rand() * n_individuals)
            while ic == ix or ic == ia or ic == ib:
                ic = int(np.random.rand() * n_individuals)
            
            a = population[ia]
            b = population[ib]
            c = population[ic]

            # random coordinate of the individual
            jr = int(np.random.rand() * n_dims)
            
            # Compute candidate individual
            y = x.copy()

            for j in range(n_dims):
                ri = np.random.rand()
                if ri < crossover_proba or j == jr:# always replace position jr, otherwise randomly decide whether to update it or not
                    y[j] = a[j] + differential_weight*(b[j] - c[j])
            
            # if candidate individual is better than current, replace current with candidate
            fy = f(y)
            if fy <= population_values[ix]:
                population[ix] = y.copy()
                population_values[ix] = fy
        
        # convergence criterion : if standard value of all function values is sufficiently small
        if np.std(np.asarray(population_values)) <= tol:
            res['n_iter'] = iter
            res['converged'] = True
            break

    # return the best
    res['x']  = population[0]
    res['fx'] = population_values[0]

    for i in range(n_individuals):
        if population_values[i] < res['fx']:
            res['x']  = population[i]
            res['fx'] = population_values[i]
    
    return res

if __name__ == '__main__':
    def ackley(X):
        x = X[0]
        y = X[1]
        return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.exp(1.0) + 20.0

    res = differential_evolution_minimize(ackley, lb=[-5., -5.], ub=[5., 5.])
    print(f'x_min     = {res["x"]}')
    print(f'f(x_min)  = {res["fx"]}')
    print(f'N iter    = {res["n_iter"]}')
    print(f'Converged = {res["converged"]}')