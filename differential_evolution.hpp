#ifndef H_differential_evolution
#define H_differential_evolution

#include <functional>
#include <Eigen/Dense>

namespace differential_evolution {

/// @brief Result of an optimization algorithm.
struct OptimizationResult {
    Eigen::VectorXd x;              //!< Minimum of the function.
    double fx;                      //!< Function value at the minimum.
    unsigned int n_iter = 0;        //!< Number of iterations.
    bool converged = false;         //!< Whether the algorithm converged.
};

/// @brief Global minimization of a function using the differential evolution algorithm.
/// @param f                        The function to minimize.
/// @param lb                       Lower bounds of the parameters.
/// @param ub                       Upper bounds of the parameters.
/// @param tol                      Tolerance on the standard deviation of the function value among all individuals of the population.
/// @param n_iter_max               Maximum number of iterations.
/// @param n_individuals            Number of individuals in the population. By default, 10 * lb.size(). Must be greater than 4.
/// @param crossover_proba          Crossover probability. Between 0 and 1.
/// @param differential_weight      Differential weight. Between 0 and 2.
/// @return The result structure containing the minimum of the function and the function value at the solution. If res.converged is false, either the algorithm did not converge, or an error occured at the beginning of the function.
OptimizationResult differential_evolution_minimize( std::function<double(Eigen::VectorXd const&)>   f,
                                                    Eigen::VectorXd const&                          lb,
                                                    Eigen::VectorXd const&                          ub,
                                                    double                                          tol                 = 1e-6,
                                                    unsigned int                                    n_iter_max          = 1000,
                                                    unsigned int                                    n_individuals       = 0,
                                                    double                                          crossover_proba     = 0.9,
                                                    double                                          differential_weight = 0.8);

}// namespace differential_evolution

#endif
