#include "differential_evolution.hpp"
#include <iostream>
#include <vector>

namespace differential_evolution {

double std_dev(Eigen::VectorXd const& vec) {
    double mean = vec.mean();
    Eigen::VectorXd deviations = (vec.array() - mean).square();
    return sqrt(deviations.sum() / vec.size());
}

unsigned int get_random_number_nmax(unsigned int nmax) {
    return (unsigned int)((Eigen::VectorXd::Random(1)[0] + 1.0)*0.5 * nmax) % nmax;
}

OptimizationResult differential_evolution_minimize(std::function<double(Eigen::VectorXd const&)> f, Eigen::VectorXd const& lb, Eigen::VectorXd const& ub, double tol, unsigned int n_iter_max, unsigned int n_individuals, double crossover_proba, double differential_weight) {
    // Result structure
    OptimizationResult res;
    res.n_iter = 0;
    res.converged = false;
    
    // Pre checks
    if(lb.size() != ub.size()) {
        std::cout << "Error : lb and ub must have the same sizes ! lb : " << lb.size() << ", ub : " << ub.size() << std::endl;
        res.x = (lb + ub)/2.0;
        return res;
    }

    unsigned int n_dims = lb.size();// Number of dimensions

    if(n_individuals < 4)           // Default value for n_individuals
        n_individuals = 10*n_dims;

    // Initialize the population
    std::vector<Eigen::VectorXd> population(n_individuals);
    Eigen::VectorXd population_values(n_individuals);
    for(unsigned int i = 0 ; i < n_individuals ; ++i) {
        Eigen::VectorXd x = Eigen::VectorXd::Random(n_dims).array()*(ub - lb).array() + lb.array();
        population[i] = x;
        population_values[i] = f(x);
    }

    // Main loop
    for(res.n_iter = 0 ; res.n_iter < n_iter_max ; ++res.n_iter) {
        // for each individual
        for(unsigned int ix = 0 ; ix < n_individuals ; ++ix) {
            Eigen::VectorXd x = population[ix];
            // pick 3 other individuals a, b, and c
            unsigned int ia = ix;
            unsigned int ib = ix;
            unsigned int ic = ix;
            while(ia == ix)
                ia = get_random_number_nmax(n_individuals);
            while(ib == ix or ib == ia)
                ib = get_random_number_nmax(n_individuals);
            while(ic == ix or ic == ia or ic == ib)
                ic = get_random_number_nmax(n_individuals);
            
            Eigen::VectorXd a = population[ia];
            Eigen::VectorXd b = population[ib];
            Eigen::VectorXd c = population[ic];

            // random coordinate of the individual
            unsigned int jr = (unsigned int)(Eigen::VectorXd::Random(1)[0] * n_dims);
            
            // Compute candidate individual
            Eigen::VectorXd y = x;

            for(unsigned int j = 0 ; j < n_dims ; ++j) {
                double ri = (Eigen::VectorXd::Random(1)[0] + 1.0)*0.5;
                if(ri < crossover_proba or j == jr)// always replace position jr, otherwise randomly decide whether to update it or not
                    y[j] = a[j] + differential_weight*(b[j] - c[j]);
            }
            
            // if candidate individual is better than current, replace current with candidate
            double fy = f(y);
            if(fy <= population_values[ix]) {
                population[ix] = y;
                population_values[ix] = fy;
            }
        }

        // convergence criterion : if standard value of all function values is sufficiently small
        if(std_dev(population_values) <= tol) {
            res.converged = true;
            break;
        }
    }

    // return the best solution
    res.x = population[0];
    res.fx = population_values[0];

    for(unsigned int i = 0 ; i < n_individuals ; ++i) {
        if(population_values[i] < res.fx) {
            res.x  = population[i];
            res.fx = population_values[i];
        }
    }

    return res;
}

} // namespace differential_evolution
