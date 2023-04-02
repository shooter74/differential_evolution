# Differential Evolution for global numerical optimization

Simple implementations of the differential evolution algorithm in C++ and python from the algorithm described in https://en.wikipedia.org/wiki/Differential_evolution .

## Dependencies
It requires the **Eigen** library for the C++ version, and the **numpy** package for python.

## Examples

### C++

The **Eigen** library is used to deal with vectors. The variable type `double` has been chosen to implement the algorithm, and the `Eigen::VectorXd` type is used for vectors.

If another type of variable needs to be used, such as a type in `boost::multiprecision`, the code can easily be adapted to use a template parameter instead of `double`.

The prototype of the function `differential_evolution_minimize` is the following :

    OptimizationResult differential_evolution_minimize(
        std::function<double(Eigen::VectorXd const&)>   f,                          // function to optimize
        Eigen::VectorXd const&                          lb,                         // lower bounds of initial domain
        Eigen::VectorXd const&                          ub,                         // upper bounds of initial domain
        double                                          tol                 = 1e-6, // tolerance on standard deviation of function values
        unsigned int                                    n_iter_max          = 1000, // maximum number of iterations
        unsigned int                                    n_individuals       = 0,    // number of individuals to use
        double                                          crossover_proba     = 0.9,  // crossover probability : in [0;1]
        double                                          differential_weight = 0.8   // differential weight : in [0;2]
    );

The file `main_diff_evolution.cpp` shows an example of usage of the function `differential_evolution_minimize` :

    #include <iostream>
    #define M_PI 3.14159265358979323846
    #include <cmath>
    #include <Eigen/Dense>
    #include "differential_evolution.hpp"

    using namespace std;

    typedef double real_t;

    real_t ackley(Eigen::VectorXd const& x) {
        real_t sum1 = 0.0;
        real_t sum2 = 0.0;
        for(unsigned int i = 0 ; i < x.size() ; ++i) {
            sum1 += x(i) * x(i);
            sum2 += std::cos(2.0 * M_PI * x(i));
        }
        return -20.0 * std::exp(-0.2 * std::sqrt(sum1 / x.size())) - std::exp(sum2 / x.size()) + 20.0 + std::exp(1.0);
    }

    int main() {
        unsigned int n_dims = 5;
        Eigen::VectorXd lb = Eigen::VectorXd::Constant(n_dims, -5.0);
        Eigen::VectorXd ub = Eigen::VectorXd::Constant(n_dims,  5.0);
        
        differential_evolution::OptimizationResult res = differential_evolution::differential_evolution_minimize(ackley, lb, ub);

        cout << "x         = " << res.x.transpose() << endl;
        cout << "f(x)      = " << res.fx << endl;
        cout << "N iter    = " << res.n_iter << endl;
        cout << "converged = " << res.converged << endl;

        return 0;
    }

# Python

The script contains an example of usage of the function `differential_evolution_minimize` :

    def ackley(X):
        x = X[0]
        y = X[1]
        return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.exp(1.0) + 20.0

    res = differential_evolution_minimize(ackley, lb=[-5., -5.], ub=[5., 5.])
    print(f'x_min     = {res["x"]}')
    print(f'f(x_min)  = {res["fx"]}')
    print(f'N iter    = {res["n_iter"]}')
    print(f'Converged = {res["converged"]}')