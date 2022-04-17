import my_plot.multivar._3D as pm3
import my_plot.multivar.contour2D as pmc
import my_plot.onevar._2D as po2
import my_scipy.multivar_optimize.minimize as smom
import my_scipy.onevar_optimize.minimize as soom
import my_numpy.decompose as npd
import my_numpy.inverse as npi
import my_numpy.solve as nps
import numpy as np
import numdifftools as nd

X=np.array([1000,897])
tol = 1.e-2
alpha = 0.1

def f(x):    
    return x[0] - x[1] + 2*(x[0]**2) + 2*x[1]*x[0] + x[1]**2

def h(x):
    return x*(x-1.5)

def r(x):
    summ=0
    for i in range (5):
        summ += x[i]**2
    return summ

X1=[]
for i in range (5):
        X1.append(i)
X1=np.array(X1)
X2=np.array([X1])

print(smom.gradient_conjugate(r,X2,tol))

#d0.T@Q@d0
# dir(nps) = ['LU','choleski', 'gaussjordan']
# dir(npi) = ['gaussjordan']
# dir(npd) = ['LU','choleski']
# dir(soom) = ['accelerated_step', 'armijo_backward', 'armijo_forward', 'dichotomous_search', 'exhaustive_search', 'fibonacci', 'fibonacci_sequence', 'fixed_step', 'golden_section', 'interval_halving']
# dir(smom) = [['sgd_with_bls_2var', 'gradient_conjugate', 'gradient_descent', 'newton','quasi_newton_dfp','sgd_2var']
# dir(po2) = ['accelerated_step', 'armijo_backward', 'armijo_forward', 'dichotomous_search', 'exhaustive_search', 'fibonacci', 'fibonacci_sequence', 'fixed_step', 'golden_section', 'interval_halving','compare_all']
# dir(pmc) = ['gradient_conjugate', 'gradient_descent', 'newton', 'quasi_newton_dfp', 'sgd', 'sgd_with_bls']
# dir(pm3) = ['gradient_conjugate', 'gradient_descent','newton','quasi_newton_dfp','sgd', 'sgd_with_bls','compare_all_time','compare_all_precision']

