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
