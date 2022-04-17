---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.8.12
  nbformat: 4
  nbformat_minor: 5
---

::: {.cell .markdown}
```{=html}
<h1><center> [S1] Projet 1: Réalisation d'un Package d'Optimisation Python </center></h1>
```
```{=html}
<h3><center> SEDJARI Yassine & MIR Zakaria </center></h3>
```
```{=html}
<center> Élèves Ingénieures en 1ère Année filière 2IA à l'ENSIAS </center>
```
`<br>`{=html} `<center>`{=html} 12 Février 2022 `</center>`{=html}
:::

::: {.cell .markdown}
## `<font color='red'>`{=html}`<u>`{=html} Plan:`</u>`{=html}`</font>`{=html} {#-plan}
:::

::: {.cell .markdown}
> ### `<u>`{=html}I. Introduction`</u>`{=html} {#i-introduction}
>
> ### `<u>`{=html}II. Package 1: my_scipy`</u>`{=html} {#ii-package-1-my_scipy}
>
> > ##### II.1 Le Sous-Package: onevar_optimize {#ii1-le-sous-package-onevar_optimize}
> >
> > > 1.  la fonction test `<br>`{=html}
> > > 2.  fixed_step `<br>`{=html}
> > > 3.  accelerated_step`<br>`{=html}
> > > 4.  exhaustive_search`<br>`{=html}
> > > 5.  dichotomous_search`<br>`{=html}
> > > 6.  interval_halving`<br>`{=html}
> > > 7.  fibonacci`<br>`{=html}
> > > 8.  golden_section`<br>`{=html}
> > > 9.  armijo_backward`<br>`{=html}
> > > 10. armijo_forward`<br>`{=html}
> >
> > ##### II.2. Le Sous-Package: multivar_optimize {#ii2-le-sous-package-multivar_optimize}
> >
> > > 1.  la fonction test `<br>`{=html}
> > > 2.  gradient_descent `<br>`{=html}
> > > 3.  gradient_conjugate`<br>`{=html}
> > > 4.  newton`<br>`{=html}
> > > 5.  quasi_newton_dfp`<br>`{=html}
> > > 6.  sgd_2var`<br>`{=html}
> > > 7.  sgd_with_bls_2var`<br>`{=html}
>
> ### `<u>`{=html}III. Package 2: my_plot`</u>`{=html} {#iii-package-2-my_plot}
>
> > ##### III.1 Le Sous-Package: onevar {#iii1-le-sous-package-onevar}
> >
> > > 1.  la fonction test `<br>`{=html}
> > > 2.  fixed_step `<br>`{=html}
> > > 3.  accelerated_step`<br>`{=html}
> > > 4.  exhaustive_search`<br>`{=html}
> > > 5.  dichotomous_search`<br>`{=html}
> > > 6.  interval_halving`<br>`{=html}
> > > 7.  fibonacci`<br>`{=html}
> > > 8.  golden_section`<br>`{=html}
> > > 9.  armijo_backward`<br>`{=html}
> > > 10. armijo_forward`<br>`{=html}
> > > 11. compare_all_time`<br>`{=html}
> > > 12. compare_all_precision`<br>`{=html}

> > ##### III.2. Le Sous-Package: multivar {#iii2-le-sous-package-multivar}
> >
> > > 1.  la fonction test `<br>`{=html}
> > > 2.  gradient_descent `<br>`{=html}
> > > 3.  gradient_conjugate`<br>`{=html}
> > > 4.  newton`<br>`{=html}
> > > 5.  quasi_newton_dfp`<br>`{=html}
> > > 6.  sgd`<br>`{=html}
> > > 7.  sgd_with_bls`<br>`{=html}
> > > 8.  compare_all_time`<br>`{=html}
> > > 9.  compare_all_precision`<br>`{=html}
>
> ### `<u>`{=html}IV. Package 3: my_numpy`</u>`{=html} {#iv-package-3-my_numpy}
>
> > #### IV.1 Le Module: inverse {#iv1-le-module-inverse}
> >
> > > 1.  la Matrice de test `<br>`{=html}
> > > 2.  gaussjordan `<br>`{=html}
> >
> > #### IV.2 Le Module: decompose {#iv2-le-module-decompose}
> >
> > > 1.  les Matrices de test `<br>`{=html}
> > > 2.  LU `<br>`{=html}
> > > 3.  choleski `<br>`{=html}
> >
> > #### IV.3 Le Module: solve {#iv3-le-module-solve}
> >
> > > 1.  les Matrices de test `<br>`{=html}
> > > 2.  gaussjordan `<br>`{=html}
> > > 3.  LU `<br>`{=html}
> > > 4.  choleski `<br>`{=html} `<br>`{=html}`<br>`{=html}
:::

::: {.cell .markdown}
## `<font color='red'>`{=html}`<u>`{=html}I Intoduction`</u>`{=html}`</font>`{=html}

`<br>`{=html}
:::

::: {.cell .markdown}
Dans le cadre de notre première année du cycle ingénieur en Intelligence
Artificielle à l'ENSIAS, il nous est proposé un projet nous permettant
de mettre en pratique toutes les connaissances acquises en matière
d\'optimisation sans contrainte pendant notre premier semestre avec le
professeur M. Naoum, au travers d'un cahier des charges ayant pour
finalité la conception et le développement d'un package d\'optimisation
à l\'aide du language de programmation python.
`<br>`{=html}`<br>`{=html}
:::

::: {.cell .markdown}
## `<font color='red'>`{=html}`<u>`{=html}II Package 1: my_scipy`</u>`{=html}`</font>`{=html}

`<br>`{=html}
:::

::: {.cell .markdown}
**my_scipy** est un package qui essaie de rassembler plusieurs packages,
modules et fonctions à usage scientifique à la manière du célèbre
package *scipy*.
:::

::: {.cell .markdown}
> ### `<font color='green'>`{=html}`<u>`{=html}II.1 Le Sous-Package: onevar_optimize`</u>`{=html}`</font>`{=html} {#ii1-le-sous-package-onevar_optimize}
:::

::: {.cell .code execution_count="1"}
``` {.python}
import my_scipy.onevar_optimize.minimize as soom
```
:::

::: {.cell .markdown}
Le package **onevar_optimize** est un sous-package de my_scipy, qui
contient le module **minimize** rassemblant les 9 fonctions de
minimisation:

1.  fixed_step
2.  accelerated_step
3.  exhaustive_search
4.  dichotomous_search
5.  interval_halving
6.  fibonacci
7.  golden_section
8.  armijo_backward
9.  armijo_forward
:::

::: {.cell .code execution_count="2" scrolled="true"}
``` {.python}
print(dir(soom))
```

::: {.output .stream .stdout}
    ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'accelerated_step', 'armijo_backward', 'armijo_forward', 'dichotomous_search', 'exhaustive_search', 'fibonacci', 'fibonacci_sequence', 'fixed_step', 'golden_section', 'interval_halving']
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.1 La fonction de test:`</u>`{=html}`</font>`{=html} {#ii11-la-fonction-de-test}
:::

::: {.cell .markdown}
En guise de simplification, on prend f une fonction dérivable qui admet
un seul minimum.
:::

::: {.cell .code execution_count="3"}
``` {.python}
def f(x):
    return x*(x-1.5)   # Analytiquement, argmin(f) = 0.75
```
:::

::: {.cell .markdown}
-   L\'intervalle d\'incertitude initial sera fixé arbitrairement à \[xs
    = -10, xf = 10\]\
-   La precision sera fixée à ε = 1.e-2\
-   Toute autre variable sera fixée par la suite
:::

::: {.cell .code execution_count="4"}
``` {.python}
xs=-10
xf=10
epsilon=1.e-2
```
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.2 fixed_step:`</u>`{=html}`</font>`{=html} {#ii12-fixed_step}
:::

::: {.cell .code execution_count="5" scrolled="true"}
``` {.python}
help(soom.fixed_step)
```

::: {.output .stream .stdout}
    Help on function fixed_step in module my_scipy.onevar_optimize.minimize:

    fixed_step(function, x0, epsilon=0.01)
        + fixed_step function takes 3 arguments:
        - f : a one variable & unimodal function 
        - x0 : the initial starting point
        - epsilon : which is the target precision
               + It returns the argmin(f)
               * the step size is fixed at epsilon
:::
:::

::: {.cell .code execution_count="6" scrolled="false"}
``` {.python}
print('x* =',soom.fixed_step(f,xs,epsilon))
```

::: {.output .stream .stdout}
    x* = 0.75
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.3 accelerated_step:`</u>`{=html}`</font>`{=html} {#ii13-accelerated_step}
:::

::: {.cell .code execution_count="7" scrolled="true"}
``` {.python}
help(soom.accelerated_step)
```

::: {.output .stream .stdout}
    Help on function accelerated_step in module my_scipy.onevar_optimize.minimize:

    accelerated_step(function, x0, epsilon=0.01)
        + accelerated_step function takes 3 arguments:
        - f : a one variable & unimodal function 
        - x0 : the initial starting point
        - epsilon : which is the target precision
               + It returns the argmin(f)
               * the step size is initialized at epsilon
:::
:::

::: {.cell .code execution_count="8" scrolled="true"}
``` {.python}
print('x* =',soom.accelerated_step(f,xs,epsilon)) 
```

::: {.output .stream .stdout}
    x* = 0.86
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.4 exhaustive_search:`</u>`{=html}`</font>`{=html} {#ii14-exhaustive_search}
:::

::: {.cell .code execution_count="9" scrolled="false"}
``` {.python}
help(soom.exhaustive_search)
```

::: {.output .stream .stdout}
    Help on function exhaustive_search in module my_scipy.onevar_optimize.minimize:

    exhaustive_search(function, xs, xf, epsilon=0.01)
        + The exhaustive_search function takes 4 arguments:
        - f : a one variable & unimodal function 
        - xs : the starting point
        - xf : the finishing point
        - epsilon : which is the target precision
               + It returns the argmin(f)
               * the step size is fixed at epsilon
:::
:::

::: {.cell .code execution_count="10"}
``` {.python}
print('x* =',soom.exhaustive_search(f,xs,xf,epsilon))
```

::: {.output .stream .stdout}
    x* = 0.75
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.5 dichotomous_search:`</u>`{=html}`</font>`{=html} {#ii15-dichotomous_search}
:::

::: {.cell .code execution_count="11" scrolled="false"}
``` {.python}
help(soom.dichotomous_search)
```

::: {.output .stream .stdout}
    Help on function dichotomous_search in module my_scipy.onevar_optimize.minimize:

    dichotomous_search(function, xs, xf, epsilon=0.01, mini_delta=0.001)
        + The dichotomous_search function takes 5 arguments:
        - f : a one variable & unimodal function 
        - xs : the starting point
        - xf : the finishing point
        - epsilon : which is the target precision
        - mini_delta : determinate the size of x_middle's neighborhood
                + It returns the argmin(f)
                ! mini_delta must be way smaller than epsilon
:::
:::

::: {.cell .code execution_count="12"}
``` {.python}
mini_delta = 1.e-3
print('x* =',soom.dichotomous_search(f,xs,xf,epsilon,mini_delta)) #ajouter x*=  et ajuster la precision selon epsilon dans le code source
```

::: {.output .stream .stdout}
    x* = 0.7494742431640624
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.6 interval_halving:`</u>`{=html}`</font>`{=html} {#ii16-interval_halving}
:::

::: {.cell .code execution_count="13" scrolled="true"}
``` {.python}
help(soom.interval_halving)
```

::: {.output .stream .stdout}
    Help on function interval_halving in module my_scipy.onevar_optimize.minimize:

    interval_halving(function, a, b, epsilon=0.01)
        + The interval_halving function takes 4 arguments:
        - f : a one variable & unimodal function f
        - a : the starting point
        - b : the finishing point
        - epsilon : which is the target precision
                + It returns the argmin(f)
:::
:::

::: {.cell .code execution_count="14"}
``` {.python}
print('x* =',soom.interval_halving(f,xs,xf,epsilon))
```

::: {.output .stream .stdout}
    x* = 0.75
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.7 fibonacci:`</u>`{=html}`</font>`{=html} {#ii17-fibonacci}
:::

::: {.cell .code execution_count="15" scrolled="false"}
``` {.python}
help(soom.fibonacci)
```

::: {.output .stream .stdout}
    Help on function fibonacci in module my_scipy.onevar_optimize.minimize:

    fibonacci(f, a, b, n=15)
        + The fibonacci function takes 4 arguments:
        - f : a one variable & unimodal function f
        - a : the starting point
        - b : the finishing point
        - n : the number of iterations to perform 
                + It returns the argmin(f)
:::
:::

::: {.cell .code execution_count="16" scrolled="true"}
``` {.python}
n=15
print('x* =',soom.fibonacci(f,xs,xf,n))  #ajouter x*=  et ajuster la precision selon epsilon dans le code source delete the last interval of uncertenty
```

::: {.output .stream .stdout}
    x* = 0.76
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.8 golden_section:`</u>`{=html}`</font>`{=html} {#ii18-golden_section}
:::

::: {.cell .code execution_count="17" scrolled="false"}
``` {.python}
help(soom.golden_section) 
```

::: {.output .stream .stdout}
    Help on function golden_section in module my_scipy.onevar_optimize.minimize:

    golden_section(f, a, b, epsilon=0.01)
        + The golden_section function takes 4 arguments:
        - f : a one variable & unimodal function f
        - a : the starting point
        - b : the finishing point
        - epsilon : which is the target precision
                + It returns the argmin(f)
:::
:::

::: {.cell .code execution_count="18"}
``` {.python}
print('x* =',soom.golden_section(f,xs,xf,epsilon))
```

::: {.output .stream .stdout}
    x* = 0.75
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.9 armijo_backward:`</u>`{=html}`</font>`{=html} {#ii19-armijo_backward}
:::

::: {.cell .code execution_count="19" scrolled="false"}
``` {.python}
help(soom.armijo_backward)
```

::: {.output .stream .stdout}
    Help on function armijo_backward in module my_scipy.onevar_optimize.minimize:

    armijo_backward(f, x0, ŋ=2, epsilon=0.01)
        + The armijo_backward (Backtracking_line_search) function takes 4 arguments:
        - f : a one variable & unimodal function f
        - x0 : the starting point
        - ŋ : the coefficient by which we divide x0 at each iteration
        - epsilon : which is the target precision
                + It returns the argmin(f)
:::
:::

::: {.cell .code execution_count="20"}
``` {.python}
ŋ=2
xs=100
print('x* =',soom.armijo_backward(f,xs,ŋ,epsilon)) #ajouter x*=  et ajuster la precision selon epsilon dans le code source delete the last interval of uncertenty
```

::: {.output .stream .stdout}
    x* = 0.78
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.1.10 armijo_forward:`</u>`{=html}`</font>`{=html} {#ii110-armijo_forward}
:::

::: {.cell .code execution_count="21" scrolled="false"}
``` {.python}
help(soom.armijo_forward)
```

::: {.output .stream .stdout}
    Help on function armijo_forward in module my_scipy.onevar_optimize.minimize:

    armijo_forward(f, x0, ŋ=2, epsilon=0.01)
        + The armijo_forward function takes 4 arguments:
        - f : a one variable & unimodal function f
        - x0 : the starting point
        - ŋ : the coefficient by which we multiply x0 at each iteration
        - epsilon : which is the target precision
                + It returns the argmin(f)
:::
:::

::: {.cell .code execution_count="22"}
``` {.python}
xs=0.1
epsilon = 0.5
ŋ=2
print('x* =',soom.armijo_forward(f,xs,ŋ,epsilon))  #ajouter x*=  et ajuster la precision selon epsilon dans le code source delete the last interval of uncertenty
```

::: {.output .stream .stdout}
    x* = 0.8
:::
:::

::: {.cell .markdown}
> ### `<font color='green'>`{=html}`<u>`{=html}II.2. Le Sous-Package: multivar_optimize`</u>`{=html}`</font>`{=html} {#ii2-le-sous-package-multivar_optimize}
:::

::: {.cell .code execution_count="23"}
``` {.python}
import my_scipy.multivar_optimize.minimize as smom
```
:::

::: {.cell .markdown}
Le package **multivar_optimize** est un sous-package de my_scipy, qui
contient le module **minimize** rassemblant les 6 fonctions de
minimisation: `<br>`{=html}

1.  gradient_descent
2.  gradient_conjugate
3.  newton
4.  quasi_newton_dfp
5.  sgd_2var
6.  sgd_with_bls_2var
:::

::: {.cell .code execution_count="24"}
``` {.python}
print(dir(smom))
```

::: {.output .stream .stdout}
    ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'diff', 'gradient_conjugate', 'gradient_descent', 'm', 'nd', 'newton', 'norm', 'np', 'quasi_newton_dfp', 'r', 'sgd_2var', 'sgd_with_bls_2var', 'spo', 'symbols']
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.2.1 La fonction de test:`</u>`{=html}`</font>`{=html} {#ii21-la-fonction-de-test}
:::

::: {.cell .code execution_count="25"}
``` {.python}
def h(x):
    return x[0] - x[1] + 2*(x[0]**2) + 2*x[1]*x[0] + x[1]**2  #argmin(g) = [-1, 1.5]
```
:::

::: {.cell .code execution_count="26"}
``` {.python}
import numpy as np
X=np.array([1000,897])
alpha=1.e-2
tol=1.e-2
```
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.2.2 gradient_descent:`</u>`{=html}`</font>`{=html} {#ii22-gradient_descent}
:::

::: {.cell .code execution_count="27" scrolled="false"}
``` {.python}
help(smom.gradient_descent)
```

::: {.output .stream .stdout}
    Help on function gradient_descent in module my_scipy.multivar_optimize.minimize:

    gradient_descent(f, X, tol=0.01, alpha=0.01)
        + gradient_descent function takes 4 arguments:
        - f : a unimodal function that takes a numpy array as an argument
        - X : a starting numpy array
        - tol : the tolerence
        - alpha : the fixed step size
               + It returns the argmin(f):  a numpy array
:::
:::

::: {.cell .code execution_count="28" scrolled="false"}
``` {.python}
print('X* =',smom.gradient_descent(h,X,tol,alpha))
```

::: {.output .stream .stdout}
    X* = [-1.01  1.51]
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.2.3 gradient_conjugate:`</u>`{=html}`</font>`{=html} {#ii23-gradient_conjugate}
:::

::: {.cell .code execution_count="29" scrolled="true"}
``` {.python}
help(smom.gradient_conjugate)
```

::: {.output .stream .stdout}
    Help on function gradient_conjugate in module my_scipy.multivar_optimize.minimize:

    gradient_conjugate(f, X, tol=0.01)
        + gradient_conjugate function takes 3 arguments:
        - f : a unimodal function that takes  a column vector as an argument
               !!! *the Hessian must be positive definite  !!! 
        - X : a starting column vector
        - tol : the tolerence
               + It returns the argmin(f): column vector
:::
:::

::: {.cell .code execution_count="30" scrolled="false"}
``` {.python}
X1=np.array([X])
print('X* =',smom.gradient_conjugate(h,X1,tol))
```

::: {.output .stream .stdout}
    X* = [[-107.38]
     [ 172.18]]
:::
:::

::: {.cell .markdown}
```{=html}
<div class="alert alert-success">
- Attention! La methode du gradient congjugé n'est par définition pas utile lorqsu'il s'agit des fonctions à deux variable , de manière générale cette methode à été conçu pour les fonctions à n variables avec n >> 1.e+6
</div>
```
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.2.4 newton:`</u>`{=html}`</font>`{=html} {#ii24-newton}
:::

::: {.cell .code execution_count="31" scrolled="true"}
``` {.python}
help(smom.newton)
```

::: {.output .stream .stdout}
    Help on function newton in module my_scipy.multivar_optimize.minimize:

    newton(f, X, tol=0.01)
        + newton function takes 3 arguments:
        - f : a unimodal function that takes  a numpy array as an argument
               *the Hessian must be invertible
        - X : a starting numpy array
        - tol : the tolerence
               + It returns the argmin(f): numpy array
:::
:::

::: {.cell .code execution_count="32"}
``` {.python}
print('X* =',smom.newton(h,X,tol))
```

::: {.output .stream .stdout}
    X* = [-1.   1.5]
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.2.5 quasi_newton_dfp:`</u>`{=html}`</font>`{=html} {#ii25-quasi_newton_dfp}
:::

::: {.cell .code execution_count="33" scrolled="true"}
``` {.python}
help(smom.quasi_newton_dfp)
```

::: {.output .stream .stdout}
    Help on function quasi_newton_dfp in module my_scipy.multivar_optimize.minimize:

    quasi_newton_dfp(f, X, tol=0.01)
        + quasi_newton_dfp takes 3 arguments:
        - f : a unimodal function that takes  a numpy array as an argument
        - X : a starting numpy array
        - tol : the tolerence
               + It returns the argmin(f): numpy array
:::
:::

::: {.cell .code execution_count="34"}
``` {.python}
print('X* =',smom.quasi_newton_dfp(h,X,tol))
```

::: {.output .stream .stdout}
    X* = [-1.   1.5]
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.2.6 sgd_2var:`</u>`{=html}`</font>`{=html} {#ii26-sgd_2var}
:::

::: {.cell .code execution_count="35" scrolled="true"}
``` {.python}
help(smom.sgd_2var)
```

::: {.output .stream .stdout}
    Help on function sgd_2var in module my_scipy.multivar_optimize.minimize:

    sgd_2var(f, X, tol=0.01, step_size=0.01)
        + stochatstic_gradient_descent_2var takes 4 arguments:
        - f : a unimodal function that takes a numpy array as an argument
        *Important the column vector must be of size 2
        - X : a starting numpy array of size 2
        - tol : the tolerence
        - step_size : de depth of de descent, it's a real value. 
               + It returns the argmin(f): a numpy array
:::
:::

::: {.cell .code execution_count="36" scrolled="true"}
``` {.python}
print('X* =',smom.sgd_2var(h,X,tol,alpha))
```

::: {.output .stream .stdout}
    X* = [-0.99  1.48]
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}II.2.7 sgd_with_bls_2var:`</u>`{=html}`</font>`{=html} {#ii27-sgd_with_bls_2var}
:::

::: {.cell .code execution_count="37" scrolled="true"}
``` {.python}
help(smom.sgd_with_bls_2var)
```

::: {.output .stream .stdout}
    Help on function sgd_with_bls_2var in module my_scipy.multivar_optimize.minimize:

    sgd_with_bls_2var(f, X, tol=0.01, initial_step_size=0.01, c=2)
        + SGD_with_BLS_2var takes 5 arguments:
        - f : a unimodal function that takes a numpy array an argument
        *Important: the numpy array must be of size 2
        - X : a starting column vector of size 2
        - tol : the tolerence
        - initial_step_size : de depth of the descent, it's a real value.
        - c: Armijo dividing coefficient c>1
               + It returns the argmin(f): a numpy array of size 2
:::
:::

::: {.cell .code execution_count="38" scrolled="false"}
``` {.python}
alpha = 100 #it must be high because of the Backtracking_Line_Search
c = 2
print('x* =',smom.sgd_with_bls_2var(h,X,tol,alpha,c)) #ajouter x*=  et ajuster la precision selon epsilon dans le code source delete the last interval of uncertenty
```

::: {.output .stream .stdout}
    x* = [-0.98  1.49]
:::
:::

::: {.cell .markdown}
`<br>`{=html}`<br>`{=html}
:::

::: {.cell .markdown}
## `<font color='red'>`{=html}`<u>`{=html}III Package 2: my_plot`</u>`{=html}`</font>`{=html}

`<br>`{=html}
:::

::: {.cell .markdown}
**my_plot** est un package qui sert à tracer visualiser et comparer les
fonctions du package *my_scipy* sous formes d\'Histogramme, de graphique
2D, de graphique 3D et sous forme de contours 2D.
:::

::: {.cell .markdown}
> ### `<font color='green'>`{=html}`<u>`{=html}III.1 Le Sous-Package: onevar`</u>`{=html}`</font>`{=html} {#iii1-le-sous-package-onevar}
:::

::: {.cell .code execution_count="39"}
``` {.python}
import my_plot.onevar._2D as po2
```
:::

::: {.cell .markdown}
Le package **onevar** est un sous-package de *my_plot*, qui contient le
module \*\*\_2D\*\* rassemblant les 9 fonctions de minimisation et 2
fonctions de comparaison: `<br>`{=html}

1.  fixed_step
2.  accelerated_step
3.  exhaustive_search
4.  dichotomous_search
5.  interval_halving
6.  fibonacci
7.  golden_section
8.  armijo_backward
9.  armijo_forward
10. compare_all_time
11. compare_all_precision
:::

::: {.cell .code execution_count="40"}
``` {.python}
print(dir(po2))
```

::: {.output .stream .stdout}
    ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_ab_', '_af_', '_as_', '_ds_', '_es_', '_fi_', '_fs_', '_fse_', '_gs_', '_ih_', 'accelerated_step', 'armijo_backward', 'armijo_forward', 'compare_all_precision', 'compare_all_time', 'dichotomous_search', 'exhaustive_search', 'fibonacci', 'fibonacci_sequence', 'fixed_step', 'golden_section', 'interval_halving', 'np', 'perf_counter', 'plt', 'spo']
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.1 La fonction de test:`</u>`{=html}`</font>`{=html} {#iii11-la-fonction-de-test}
:::

::: {.cell .markdown}
En guise de simplification, on prend f un fonction dérivable qui admet
un seul minimum.
:::

::: {.cell .code execution_count="41"}
``` {.python}
def f(x):
    return x*(x-1.5)   # Analytiquement, argmin(f) = 0.75
```
:::

::: {.cell .markdown}
-   L\'intervalle d\'incertitude initial sera fixé arbitrairement à \[xs
    = -10, xf = 10\]\
-   La precision sera fixée à ε = 1.e-2\
-   Toute autre variable sera fixée par la suite
:::

::: {.cell .code execution_count="42"}
``` {.python}
xs=-10
xf=10
epsilon=1.e-2
```
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.2 fixed_step:`</u>`{=html}`</font>`{=html} {#iii12-fixed_step}
:::

::: {.cell .code execution_count="43" scrolled="false"}
``` {.python}
help(po2.fixed_step)
```

::: {.output .stream .stdout}
    Help on function fixed_step in module my_plot.onevar._2D:

    fixed_step(f, x0, epsilon=0.01)
        + fixed_step function takes 3 arguments:
        - f : a one variable & unimodal function 
        - x0 : the initial starting point
        - epsilon : which is the target precision
               + It returns the argmin(f) & the 2D plot of the search
               * the step size is fixed at epsilon
:::
:::

::: {.cell .code execution_count="44" scrolled="true"}
``` {.python}
po2.fixed_step(f,xs,epsilon)
```

::: {.output .stream .stdout}
    x* =  0.7549999999998316
    Le nombre de points parcourus avant d'arriver au minimum est  1077
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/051c1df6a27638f735672a3e252959a652d7d778.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.3 accelerated_step:`</u>`{=html}`</font>`{=html} {#iii13-accelerated_step}
:::

::: {.cell .code execution_count="45" scrolled="true"}
``` {.python}
help(po2.accelerated_step)
```

::: {.output .stream .stdout}
    Help on function accelerated_step in module my_plot.onevar._2D:

    accelerated_step(f, x0, epsilon=0.01)
        + accelerated_step function takes 3 arguments:
        - f : a one variable & unimodal function 
        - x0 : the initial starting point
        - epsilon : which is the target precision
               + It returns the argmin(f) & the 2D plot of the search
               * the step size is initialized at epsilon
:::
:::

::: {.cell .code execution_count="46" scrolled="true"}
``` {.python}
po2.accelerated_step(f,xs,epsilon)
```

::: {.output .stream .stdout}
    x* =  0.8649999999999997
    Le nombre de points parcourus avant d'arriver au minimum est  18
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/7dea9121fe848b1fc35d230320cd817b77f234b4.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.4 exhaustive_search:`</u>`{=html}`</font>`{=html} {#iii14-exhaustive_search}
:::

::: {.cell .code execution_count="47" scrolled="false"}
``` {.python}
help(po2.exhaustive_search)
```

::: {.output .stream .stdout}
    Help on function exhaustive_search in module my_plot.onevar._2D:

    exhaustive_search(f, xs, xf, epsilon=0.01)
        + The exhaustive_search function takes 4 arguments:
        - f : a one variable & unimodal function 
        - xs : the starting point
        - xf : the finishing point
        - epsilon : which is the target precision
               + It returns the argmin(f) & the 2D plot of the search
               * the step size is fixed at epsilon
:::
:::

::: {.cell .code execution_count="48" scrolled="true"}
``` {.python}
po2.exhaustive_search(f,xs,xf,epsilon)
```

::: {.output .stream .stdout}
    x* =  0.75
    Le nombre de points parcourus avant d'arriver au minimum est  2003
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/60dac81c27447bd6631c8d7fdeefc3d4b4ea1b3b.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.5 dichotomous_search:`</u>`{=html}`</font>`{=html} {#iii15-dichotomous_search}
:::

::: {.cell .code execution_count="49" scrolled="false"}
``` {.python}
help(po2.dichotomous_search)
```

::: {.output .stream .stdout}
    Help on function dichotomous_search in module my_plot.onevar._2D:

    dichotomous_search(f, xs, xf, epsilon=0.01, mini_delta=0.001)
        + The dichotomous_search function takes 5 arguments:
        - f : a one variable & unimodal function 
        - xs : the starting point
        - xf : the finishing point
        - epsilon : which is the target precision
        - mini_delta : determinate the size of x_middle's neighborhood
                + It returns the argmin(f) & the 2D plot of the search
                ! mini_delta must be way smaller than epsilon
:::
:::

::: {.cell .code execution_count="50" scrolled="true"}
``` {.python}
mini_delta = 1.e-3
po2.dichotomous_search(f,xs,xf,epsilon,mini_delta)
```

::: {.output .stream .stdout}
    x* =  0.7494742431640624
    Le nombre de points parcourus avant d'arriver au minimum est  14
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/39cdc1f5d6702c2d44e794a13fe757023e4f6247.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.6 interval_halving:`</u>`{=html}`</font>`{=html} {#iii16-interval_halving}
:::

::: {.cell .code execution_count="51" scrolled="true"}
``` {.python}
help(po2.interval_halving)
```

::: {.output .stream .stdout}
    Help on function interval_halving in module my_plot.onevar._2D:

    interval_halving(f, a, b, epsilon=0.01)
        + The interval_halving function takes 4 arguments:
        - f : a one variable & unimodal function f
        - a : the starting point
        - b : the finishing point
        - epsilon : which is the target precision
                + It returns the argmin(f) & the 2D plot of the search
:::
:::

::: {.cell .code execution_count="52" scrolled="false"}
``` {.python}
po2.interval_halving(f,xs,xf,epsilon)
```

::: {.output .stream .stdout}
    x* =  0.751953125
    Le nombre de points parcourus avant d'arriver au minimum est  11
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/1d9d4ae84b41d225816151803cf0a35bcb3e66cd.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.7 fibonacci:`</u>`{=html}`</font>`{=html} {#iii17-fibonacci}
:::

::: {.cell .code execution_count="53" scrolled="false"}
``` {.python}
help(po2.fibonacci)
```

::: {.output .stream .stdout}
    Help on function fibonacci in module my_plot.onevar._2D:

    fibonacci(f, a, b, n=15)
        + The fibonacci function takes 4 arguments:
        - f : a one variable & unimodal function f
        - a : the starting point
        - b : the finishing point
        - n : the number of iterations to perform 
                + It returns the argmin(f) & the 2D plot of the search
:::
:::

::: {.cell .code execution_count="54" scrolled="true"}
``` {.python}
n=15
po2.fibonacci(f,xs,xf,n)
```

::: {.output .stream .stdout}
    x* =  0.7598784194528874
    Le nombre de points parcourus avant d'arriver au minimum est  15
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/ff2c51b66e8351be7105479a25eedd1c4caf5854.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.8 golden_section:`</u>`{=html}`</font>`{=html} {#iii18-golden_section}
:::

::: {.cell .code execution_count="55" scrolled="false"}
``` {.python}
help(po2.golden_section) 
```

::: {.output .stream .stdout}
    Help on function golden_section in module my_plot.onevar._2D:

    golden_section(f, a, b, epsilon=0.01)
        + The golden_section function takes 4 arguments:
        - f : a one variable & unimodal function f
        - a : the starting point
        - b : the finishing point
        - epsilon : which is the target precision
                + It returns the argmin(f) & the 2D plot of the search
:::
:::

::: {.cell .code execution_count="56"}
``` {.python}
po2.golden_section(f,xs,xf,epsilon)
```

::: {.output .stream .stdout}
    x* =  0.7481556335102038
    Le nombre de points parcourus avant d'arriver au minimum est  17
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/f4eedc0e512131b7229803c3eb969bd19fc4ed43.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.9 armijo_backward:`</u>`{=html}`</font>`{=html} {#iii19-armijo_backward}
:::

::: {.cell .code execution_count="57" scrolled="false"}
``` {.python}
help(po2.armijo_backward)
```

::: {.output .stream .stdout}
    Help on function armijo_backward in module my_plot.onevar._2D:

    armijo_backward(f, x0, ŋ=2, epsilon=0.01)
        + The armijo_backward (Backtracking_line_search) function takes 4 arguments:
        - f : a one variable & unimodal function f
        - x0 : the starting point
        - ŋ : the coefficient by which we divide x0 at each iteration
        - epsilon : which is the target precision
                + It returns the argmin(f) & the 2D plot of the search
:::
:::

::: {.cell .code execution_count="58" scrolled="true"}
``` {.python}
ŋ=2
xs=100
po2.armijo_backward(f,xs,ŋ,epsilon)
```

::: {.output .stream .stdout}
    x* =  0.78125
    Le nombre de points parcourus avant d'arriver au minimum est  8
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/79a4b7d384e553ae8674637f4ee6761c340e8e47.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.10 armijo_forward:`</u>`{=html}`</font>`{=html} {#iii110-armijo_forward}
:::

::: {.cell .code execution_count="59" scrolled="false"}
``` {.python}
help(po2.armijo_forward)
```

::: {.output .stream .stdout}
    Help on function armijo_forward in module my_plot.onevar._2D:

    armijo_forward(f, x0, ŋ=2, epsilon=0.01)
        + The armijo_forward function takes 4 arguments:
        - f : a one variable & unimodal function f
        - x0 : the starting point
        - ŋ : the coefficient by which we multiply x0 at each iteration
        - epsilon : which is the target precision
                + It returns the argmin(f) & the 2D plot of the search
:::
:::

::: {.cell .code execution_count="60" scrolled="true"}
``` {.python}
xs=0.1
epsilon = 0.1
ŋ=2
po2.armijo_forward(f,xs,ŋ,epsilon)
```

::: {.output .stream .stdout}
    x* =  0.8
    Le nombre de points parcourus avant d'arriver au minimum est  6
    Le plot apparaitra dans quelques secondes, merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/a5c88d5ca11647168b15ca16455caeed9933a07b.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.10 compare_all_time:`</u>`{=html}`</font>`{=html} {#iii110-compare_all_time}
:::

::: {.cell .code execution_count="61"}
``` {.python}
help(po2.compare_all_time)
```

::: {.output .stream .stdout}
    Help on function compare_all_time in module my_plot.onevar._2D:

    compare_all_time(f, xs, xf, epsilon, mini_delta_dichotomous, n_fibo, ŋ_armijo, xs_armijo_forward, xs_armijo_backward)
:::
:::

::: {.cell .code execution_count="62"}
``` {.python}
po2.compare_all_time(f,0,2,1.e-2,1.e-3,10,2,0.1,100)
```

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/4ba934b110d61ce29f58ad404598909e100e4035.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.1.10 compare_all_precision:`</u>`{=html}`</font>`{=html} {#iii110-compare_all_precision}
:::

::: {.cell .code execution_count="63"}
``` {.python}
help(po2.compare_all_precision)
```

::: {.output .stream .stdout}
    Help on function compare_all_precision in module my_plot.onevar._2D:

    compare_all_precision(f, xs, xf, epsilon, mini_delta_dichotomous, n_fibo, ŋ_armijo, xs_armijo_forward, xs_armijo_backward)
:::
:::

::: {.cell .code execution_count="64"}
``` {.python}
po2.compare_all_precision(f,0,2,1.e-2,1.e-3,10,2,0.1,100)
```

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/ace8f41ffefdf0456ea8cfeafb9fd6d2079cc7a1.png)
:::
:::

::: {.cell .markdown}
> ### `<font color='green'>`{=html}`<u>`{=html}III.2. Le Sous-Package: multivar`</u>`{=html}`</font>`{=html} {#iii2-le-sous-package-multivar}
:::

::: {.cell .code execution_count="65"}
``` {.python}
import my_plot.multivar._3D as pm3
import my_plot.multivar.contour2D as pmc
```
:::

::: {.cell .markdown}
Le package **multivar** est un sous-package de *my_plot*, qui contient
le module \*\*\_3D\*\* et le module **contour2D** rassemblant chaqu\'un
les 6 fonctions de minimisation: `<br>`{=html}

1.  gradient_descent
2.  gradient_conjugate
3.  newton
4.  quasi_newton_dfp
5.  sgd
6.  sgd_with_bls
:::

::: {.cell .code execution_count="66" scrolled="false"}
``` {.python}
print(dir(pm3))
```

::: {.output .stream .stdout}
    ['Axes3D', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_gc_', '_gd_', '_n_', '_qndfp_', '_sgd_', '_sgdbls_', 'compare_all_precision', 'compare_all_time', 'diff', 'gradient_conjugate', 'gradient_descent', 'm', 'mplot3d', 'nd', 'newton', 'norm', 'np', 'perf_counter', 'plt', 'quasi_newton_dfp', 'r', 'sgd', 'sgd_with_bls', 'spo', 'symbols', 'time']
:::
:::

::: {.cell .code execution_count="67"}
``` {.python}
print(dir(pmc))
```

::: {.output .stream .stdout}
    ['Axes3D', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'diff', 'gradient_conjugate', 'gradient_descent', 'm', 'mplot3d', 'nd', 'newton', 'norm', 'np', 'plt', 'quasi_newton_dfp', 'r', 'sgd', 'sgd_with_bls', 'spo', 'symbols', 'time']
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.1 La fonction de test:`</u>`{=html}`</font>`{=html} {#iii21-la-fonction-de-test}
:::

::: {.cell .code execution_count="68"}
``` {.python}
def h(x):
    return x[0] - x[1] + 2*(x[0]**2) + 2*x[1]*x[0] + x[1]**2
```
:::

::: {.cell .code execution_count="69"}
``` {.python}
import numpy as np
X=[1000,897]
alpha=1.e-2
tol=1.e-2
```
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.2 gradient_descent:`</u>`{=html}`</font>`{=html} {#iii22-gradient_descent}
:::

::: {.cell .code execution_count="70" scrolled="true"}
``` {.python}
help(pmc.gradient_descent)
```

::: {.output .stream .stdout}
    Help on function gradient_descent in module my_plot.multivar.contour2D:

    gradient_descent(f, X, tol=0.01, alpha=0.01)
        + gradient_descent function takes 3 arguments:
        - f : a unimodal function that takes a numpy array as an argument
        - X : a starting numpy array
        - tol : the tolerence
        - alpha : the depth of the descent, it's a real value.
               + It returns the argmin(f): a numpy array of size 2 & the 2D contour plot of the search
:::
:::

::: {.cell .code execution_count="71" scrolled="false"}
``` {.python}
help(pm3.gradient_descent)
```

::: {.output .stream .stdout}
    Help on function gradient_descent in module my_plot.multivar._3D:

    gradient_descent(f, X, tol=0.01, alpha=0.01)
        + gradient_descent function takes 4 arguments:
        - f : a unimodal function that takes a numpy array as an argument
        - X : a starting numpy array
        - tol : the tolerence
        - alpha : the depth of the descent, it's a real value.
               + It returns the argmin(f): a numpy array of size 2 & the 3D plot of the search
:::
:::

::: {.cell .code execution_count="72" scrolled="false"}
``` {.python}
pmc.gradient_descent(h,X,tol,alpha)
```

::: {.output .stream .stdout}
    Y* =  [-1.01  1.51]
    Le nombre de points parcourus lors de la descente est  1279
    Le plot contour2D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/e361a0def4d04f0dfd9afcd56813602dacabe0d6.png)
:::
:::

::: {.cell .code execution_count="73" scrolled="true"}
``` {.python}
pm3.gradient_descent(h,X,tol,alpha)
```

::: {.output .stream .stdout}
    Y* =  [-1.01  1.51]
    Le nombre de points parcourus lors de la descente est  1279
    Le plot 3D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/6f3aeed40b98a1d0831aefb5bfc125e600b08a64.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.3 gradient_conjugate:`</u>`{=html}`</font>`{=html} {#iii23-gradient_conjugate}
:::

::: {.cell .code execution_count="74" scrolled="false"}
``` {.python}
help(pmc.gradient_conjugate)
```

::: {.output .stream .stdout}
    Help on function gradient_conjugate in module my_plot.multivar.contour2D:

    gradient_conjugate(f, X, tol=0.01)
        + gradient_conjugate function takes 3 arguments:
        - f : a unimodal function that takes  a column vector as an argument
               *the Hessian must be positive definite 
        - X : a starting column vector
        - tol : the tolerence
               + It returns the argmin(f): a column vector  of size 2 & the 2D contour plot of the search
:::
:::

::: {.cell .code execution_count="75" scrolled="false"}
``` {.python}
help(pm3.gradient_conjugate)
```

::: {.output .stream .stdout}
    Help on function gradient_conjugate in module my_plot.multivar._3D:

    gradient_conjugate(f, X, tol=0.01)
        + gradient_conjugate function takes 3 arguments:
        - f : a unimodal function that takes  a column vector as an argument
               *the Hessian must be positive definite 
        - X : a starting column vector
        - tol : the tolerence
               + It returns the argmin(f): a column vector of size 2 & the 3D plot of the search
:::
:::

::: {.cell .code execution_count="76" scrolled="false"}
``` {.python}
pmc.gradient_conjugate(h,X,tol)
```

::: {.output .stream .stdout}
    Y* =  [-107.38  172.18]
    Le nombre de points parcourus lors de la descente est  2
    Le plot contour2D apparaitra dans quelques secondes ...
    Merci de patienter !
    !!!!! This  method is not accurate in dimension 3 because it computes only one iteration, it's effective when the dimensoin n is very high > 1.e6
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/9b843b6583de9d29d2326c4b9a7d3a58ed3cf544.png)
:::
:::

::: {.cell .markdown}
```{=html}
<div class="alert alert-success">
WARNING! This  method is not accurate in dimension 3 because it computes only one iteration, it's effective when the dimension n is very high >> 1.e6
</div>
```
:::

::: {.cell .code execution_count="77" scrolled="true"}
``` {.python}
pm3.gradient_conjugate(h,X,tol)
```

::: {.output .stream .stdout}
    Y* =  [-107.38  172.18]
    Le nombre de points parcourus lors de la descente est  2
    Le plot 3D apparaitra dans quelques secondes ...
    Merci de patienter !
    !!!!! This  method is not accurate in dimension 3 because it computes only one iteration, it's effective when the dimensoin n is very high > 1.e6
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/ad6c4b3b28698634aef8666703788fc3f83df29a.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.4 newton:`</u>`{=html}`</font>`{=html} {#iii24-newton}
:::

::: {.cell .code execution_count="78" scrolled="false"}
``` {.python}
help(pmc.newton)
```

::: {.output .stream .stdout}
    Help on function newton in module my_plot.multivar.contour2D:

    newton(f, X, tol=0.01)
        + newton function takes 3 arguments:
        - f : a unimodal function that takes  a numpy array as an argument
               *the Hessian must be invertible
        - X : a starting numpy array
        - tol : the tolerence
               + It returns the argmin(f): a column vector  of size 2 & the 2D contour plot of the search
:::
:::

::: {.cell .code execution_count="79" scrolled="false"}
``` {.python}
help(pm3.newton)
```

::: {.output .stream .stdout}
    Help on function newton in module my_plot.multivar._3D:

    newton(f, X, tol=0.01)
        + newton function takes 3 arguments:
        - f : a unimodal function that takes  a numpy array as an argument
               *the Hessian must be invertible
        - X : a starting numpy array
        - tol : the tolerence
               + It returns the argmin(f): a column vector of size 2 & the 3D plot of the search
:::
:::

::: {.cell .code execution_count="80" scrolled="true"}
``` {.python}
pmc.newton(h,X,tol)
```

::: {.output .stream .stdout}
    Y* =  [-1.   1.5]
    Le nombre de points parcourus lors de la descente est  2
    Le plot contour2D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/4c047421f68cf8fc3540aa5c2b2bba6b18f3b653.png)
:::
:::

::: {.cell .code execution_count="81" scrolled="true"}
``` {.python}
pm3.newton(h,X,tol)
```

::: {.output .stream .stdout}
    Y* =  [-1.   1.5]
    Le nombre de points parcourus lors de la descente est  2
    Le plot 3D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/d48ce83f99b5b05fa8668cff7f4f77a8096c337c.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.5 quasi_newton_dfp:`</u>`{=html}`</font>`{=html} {#iii25-quasi_newton_dfp}
:::

::: {.cell .code execution_count="82" scrolled="true"}
``` {.python}
help(pmc.quasi_newton_dfp)
```

::: {.output .stream .stdout}
    Help on function quasi_newton_dfp in module my_plot.multivar.contour2D:

    quasi_newton_dfp(f, X, tol=0.01)
        + quasi_newton_dfp takes 3 arguments:
        - f : a unimodal function that takes  a numpy array as an argument
        - X : a starting numpy array
        - tol : the tolerence
               + It returns the argmin(f): a column vector  of size 2 & the 2D contour plot of the search
:::
:::

::: {.cell .code execution_count="83" scrolled="true"}
``` {.python}
help(pm3.quasi_newton_dfp)
```

::: {.output .stream .stdout}
    Help on function quasi_newton_dfp in module my_plot.multivar._3D:

    quasi_newton_dfp(f, X, tol=0.01)
        + quasi_newton_dfp takes 3 arguments:
        - f : a unimodal function that takes  a numpy array as an argument
        - X : a starting numpy array
        - tol : the tolerence
               + It returns the argmin(f): a column vector of size 2 & the 3D plot of the search
:::
:::

::: {.cell .code execution_count="84"}
``` {.python}
pmc.quasi_newton_dfp(h,X,tol)
```

::: {.output .stream .stdout}
    Y* =  [-1.   1.5]
    Le nombre de points parcourus lors de la descente est  3
    Le plot contour2D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/bee2ecaef62a569bc1bc3fb64ce6b730bcb44e64.png)
:::
:::

::: {.cell .code execution_count="85" scrolled="false"}
``` {.python}
pm3.quasi_newton_dfp(h,X,tol)
```

::: {.output .stream .stdout}
    Y* =  [-1.   1.5]
    Le nombre de points parcourus lors de la descente est  3
    Le plot 3D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/1a8847772d0a8d2ad79945b072e0d7b4287f06b8.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.6 sgd:`</u>`{=html}`</font>`{=html} {#iii26-sgd}
:::

::: {.cell .code execution_count="86" scrolled="true"}
``` {.python}
help(pmc.sgd)
```

::: {.output .stream .stdout}
    Help on function sgd in module my_plot.multivar.contour2D:

    sgd(f, X, tol=0.01, step_size=0.01)
        + sgd takes 5 arguments:
        - f : a unimodal function that takes a numpy array an argument
        *Important: the numpy array must be of size 2
        - X : a starting column vector of size 2
        - tol : the tolerence
        - initial_step_size : the depth of the descent, it's a real value.
        - c: Armijo dividing coefficient c>1
               + It returns the argmin(f): a numpy array of size 2 & the 2D contour plot of the search
:::
:::

::: {.cell .code execution_count="87" scrolled="true"}
``` {.python}
help(pm3.sgd)
```

::: {.output .stream .stdout}
    Help on function sgd in module my_plot.multivar._3D:

    sgd(f, X, tol, step_size=0.01)
        + sgd takes 5 arguments:
        - f : a unimodal function that takes a numpy array an argument
        *Important: the numpy array must be of size 2
        - X : a starting column vector of size 2
        - tol : the tolerence
        - initial_step_size : the depth of the descent, it's a real value.
        - c: Armijo dividing coefficient c>1
               + It returns the argmin(f): a numpy array of size 2 & the 3D plot of the search
:::
:::

::: {.cell .code execution_count="88" scrolled="true"}
``` {.python}
pmc.sgd(h,X,tol,alpha)
```

::: {.output .stream .stdout}
    X* =  [-1.01  1.52]
    Le nombre de points parcourus lors de la descente est  2513
    Le plot contour2D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/61572cb13fde135d4fade46aab7d820f2386ed6e.png)
:::
:::

::: {.cell .code execution_count="89" scrolled="false"}
``` {.python}
pm3.sgd(h,X,tol,alpha)
```

::: {.output .stream .stdout}
    Y* =  [-1.01  1.52]
    Le plot 3D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/77c78233e18f159f2d9142159db9b85032daef0d.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.7 sgd_with_bls:`</u>`{=html}`</font>`{=html} {#iii27-sgd_with_bls}
:::

::: {.cell .code execution_count="90" scrolled="true"}
``` {.python}
help(pmc.sgd_with_bls)
```

::: {.output .stream .stdout}
    Help on function sgd_with_bls in module my_plot.multivar.contour2D:

    sgd_with_bls(f, X, tol=0.01, initial_step_size=0.01, c=2)
        + sgd_with_bls takes 5 arguments:
        - f : a unimodal function that takes a numpy array an argument
        *Important: the numpy array must be of size 2
        - X : a starting column vector of size 2
        - tol : the tolerence
        - initial_step_size : de depth of the descent, it's a real value.
        - c: Armijo dividing coefficient c>1
               + It returns the argmin(f): a numpy array of size 2 & the 2D contour plot of the search
:::
:::

::: {.cell .code execution_count="91" scrolled="true"}
``` {.python}
help(pm3.sgd_with_bls)
```

::: {.output .stream .stdout}
    Help on function sgd_with_bls in module my_plot.multivar._3D:

    sgd_with_bls(f, X, tol=0.01, initial_step_size=0.01, c=2)
        + sgd_with_bls takes 5 arguments:
        - f : a unimodal function that takes a numpy array an argument
        *Important: the numpy array must be of size 2
        - X : a starting column vector of size 2
        - tol : the tolerence
        - initial_step_size : de depth of the descent, it's a real value.
        - c: Armijo dividing coefficient c>1
               + It returns the argmin(f): a numpy array of size 2 & the 3D plot of the search
:::
:::

::: {.cell .code execution_count="92" scrolled="true"}
``` {.python}
alpha = 100 #it must be high because of BLS
c = 2
pmc.sgd_with_bls(h,X,tol,alpha,c) 
```

::: {.output .stream .stdout}
    X* =  [-1.01  1.5 ]
    Le nombre de points parcourus lors de la descente est  67
    Le plot contour2D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/fdfbb1525660df60364cc7eb06fb02b58b60471c.png)
:::
:::

::: {.cell .code execution_count="93" scrolled="true"}
``` {.python}
alpha = 100 #it must be high because of BLS
c = 2
pm3.sgd_with_bls(h,X,tol,alpha,c)
```

::: {.output .stream .stdout}
    Y* =  [-1.   1.5]
    Le nombre de points parcourus lors de la descente est  80
    Le plot 3D apparaitra dans quelques secondes ...
    Merci de patienter !
:::

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/081d54546745824d17fd64b8d67bd4097cf11b2e.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.7 compare_all_time:`</u>`{=html}`</font>`{=html} {#iii27-compare_all_time}
:::

::: {.cell .code execution_count="94"}
``` {.python}
help(pm3.compare_all_time)
```

::: {.output .stream .stdout}
    Help on function compare_all_time in module my_plot.multivar._3D:

    compare_all_time(f, X, tol, alpha, xstart_bls, n_bls=2)
:::
:::

::: {.cell .code execution_count="95" scrolled="true"}
``` {.python}
pm3.compare_all_time(h,X,1.e-2,1.e-1,100,2)
```

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/50d6e59caf119abe411ef75d3aba6a2ed6e7fb77.png)
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}III.2.7 compare_all_precision:`</u>`{=html}`</font>`{=html} {#iii27-compare_all_precision}
:::

::: {.cell .code execution_count="96"}
``` {.python}
help(pm3.compare_all_precision)
```

::: {.output .stream .stdout}
    Help on function compare_all_precision in module my_plot.multivar._3D:

    compare_all_precision(f, X, tol, alpha, xstart_bls, n_bls=2)
:::
:::

::: {.cell .code execution_count="97" scrolled="true"}
``` {.python}
pm3.compare_all_precision(h,X,1.e-2,1.e-1,100,2)
```

::: {.output .display_data}
![](vertopal_78f77000ff8148afa0dcaa38e4f93608/a74596ecceab383518944f7711c30496a8a1bb16.png)
:::
:::

::: {.cell .markdown}
`<br>`{=html}`<br>`{=html}
:::

::: {.cell .markdown}
## `<font color='red'>`{=html}`<u>`{=html}IV Package 3: my_numpy`</u>`{=html}`</font>`{=html}

`<br>`{=html}
:::

::: {.cell .markdown}
**my_numpy** est un package qui essaie de rassembler plusieurs autres
packages, modules et fonctions destinée à manipuler des matrices à la
manière du célèbre package *numpy*. il contient les 3 modules
**inverse**, **decompose** et **solve**.
:::

::: {.cell .markdown}
> ### `<font color='green'>`{=html}`<u>`{=html}IV.1 Le Module: inverse`</u>`{=html}`</font>`{=html} {#iv1-le-module-inverse}
:::

::: {.cell .code execution_count="98"}
``` {.python}
import my_numpy.inverse as npi
```
:::

::: {.cell .markdown}
le module **inverse** contient une seule fonction : **gaussjordan**
:::

::: {.cell .code execution_count="99" scrolled="true"}
``` {.python}
print(dir(npi))
```

::: {.output .stream .stdout}
    ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'gaussjordan', 'm', 'np']
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.1.1 La Matrice de test:`</u>`{=html}`</font>`{=html} {#iv11-la-matrice-de-test}
:::

::: {.cell .markdown}
En guise de simplification, on prend A une matrice carré d\'ordre 3
:::

::: {.cell .code execution_count="100"}
``` {.python}
import numpy as np
A = np.array([[1.,2.,3.],[0.,1.,4.],[5.,6.,0.]])
```
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.1.2 gaussjordan:`</u>`{=html}`</font>`{=html} {#iv12-gaussjordan}
:::

::: {.cell .code execution_count="101" scrolled="false"}
``` {.python}
help(npi.gaussjordan)
```

::: {.output .stream .stdout}
    Help on function gaussjordan in module my_numpy.inverse:

    gaussjordan(A)
        gaussjordan(A) takes a square matrix A as an argument,
        
            if A is invertable, it returns it's inverse using Gauss-Jordan method
            otherwise it returns an error message.
:::
:::

::: {.cell .code execution_count="102"}
``` {.python}
A_1=npi.gaussjordan(A.copy())
I=A@A_1
I=np.around(I,1)
print('A_1 =\n\n',A_1)
print('\nA_1*A =\n\n',I)
```

::: {.output .stream .stdout}
    A_1 =

     [[-24.  18.   5.]
     [ 20. -15.  -4.]
     [ -5.   4.   1.]]

    A_1*A =

     [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
:::
:::

::: {.cell .markdown}
> ### `<font color='green'>`{=html}`<u>`{=html}IV.2 Le Module: decompose`</u>`{=html}`</font>`{=html} {#iv2-le-module-decompose}
:::

::: {.cell .code execution_count="103"}
``` {.python}
import my_numpy.decompose as npd
```
:::

::: {.cell .markdown}
Le module **decompose** contient deux fonctions :

1.  LU
2.  choleski
:::

::: {.cell .code execution_count="104" scrolled="true"}
``` {.python}
print(dir(npd))
```

::: {.output .stream .stdout}
    ['LU', '__builtins__', '__cached__', '__doc__', '__file__', '__inv__', '__loader__', '__name__', '__package__', '__spec__', 'choleski', 'is_pos_def', 'm', 'np']
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.2.1 La Matrice de test:`</u>`{=html}`</font>`{=html} {#iv21-la-matrice-de-test}
:::

::: {.cell .markdown}
En guise de simplification, on prend A une matrice carré d\'ordre 3
:::

::: {.cell .code execution_count="105"}
``` {.python}
import numpy as np
A = np.array([[1.,2.,3.],[0.,1.,4.],[5.,6.,0.]])      #A n'est pas definie positive 
B = np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])  #B est definie positive

Y=np.array([45,-78,95])                               #vecteur colonne choisie au hasard
```
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.2.2 LU:`</u>`{=html}`</font>`{=html} {#iv22-lu}
:::

::: {.cell .code execution_count="106" scrolled="false"}
``` {.python}
help(npd.LU)
```

::: {.output .stream .stdout}
    Help on function LU in module my_numpy.decompose:

    LU(A)
        LU(A) takes a square matrix A as an argument and returns a tuple of 3 square matrices (L,U,P) such that:
        
            A = P@L@U , where:
                                P is a permutation matrix
                                L is a lower triangular matrix
                                U is an upper tiangular matrix
:::
:::

::: {.cell .code execution_count="107"}
``` {.python}
L,U,P=npd.LU(A)
print("P =\n",P,"\n\nL =\n",L,"\n\nU =\n",U)
print("\n",A==P@L@U)
```

::: {.output .stream .stdout}
    P =
     [[0. 0. 1.]
     [0. 1. 0.]
     [1. 0. 0.]] 

    L =
     [[1.  0.  0. ]
     [0.  1.  0. ]
     [0.2 0.8 1. ]] 

    U =
     [[ 5.   6.   0. ]
     [ 0.   1.   4. ]
     [ 0.   0.  -0.2]]

     [[ True  True  True]
     [ True  True  True]
     [ True  True  True]]
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.2.3 choleski`</u>`{=html}`</font>`{=html} {#iv23-choleski}
:::

::: {.cell .code execution_count="108" scrolled="false"}
``` {.python}
help(npd.choleski)
```

::: {.output .stream .stdout}
    Help on function choleski in module my_numpy.decompose:

    choleski(A)
        choleski(A) takes a square matrix A as an argument and
        
            if A is positive definite it returns a square matrix L such that: A = L@(L.T) , where L is a lower triangular matrix.
        
            otherwise it returns an error message
:::
:::

::: {.cell .code execution_count="109" scrolled="true"}
``` {.python}
L=npd.choleski(A)          # A is not positive definite
print(L)
print("--------------------------------------------------")
L=npd.choleski(B)          # B is positive definite 
print('L =\n',L,'\n')

C=np.around(L@(L.T),1)
print('B = L@(L.T) \n\n',B==C)

```

::: {.output .stream .stdout}
    A must be positive definite !
    None
    --------------------------------------------------
    L =
     [[ 1.41421356  0.          0.        ]
     [-0.70710678  1.22474487  0.        ]
     [ 0.         -0.81649658  1.15470054]] 

    B = L@(L.T) 

     [[ True  True  True]
     [ True  True  True]
     [ True  True  True]]
:::
:::

::: {.cell .markdown}
> ### `<font color='green'>`{=html}`<u>`{=html}IV.3 Le Module: solve`</u>`{=html}`</font>`{=html} {#iv3-le-module-solve}
:::

::: {.cell .code execution_count="110"}
``` {.python}
import my_numpy.solve as nps
```
:::

::: {.cell .markdown}
Le module **solve** contient trois fonctions :

1.  gaussjordan
2.  LU
3.  choleski
:::

::: {.cell .code execution_count="111" scrolled="true"}
``` {.python}
print(dir(nps))
```

::: {.output .stream .stdout}
    ['LU', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_cho_', '_inv_', '_is_cho_', '_lu_', 'choleski', 'gaussjordan', 'm', 'np']
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.3.1 La Matrice de test:`</u>`{=html}`</font>`{=html} {#iv31-la-matrice-de-test}
:::

::: {.cell .markdown}
-   En guise de simplification, on prend A et B des matrices carrées
    d\'ordre 3
:::

::: {.cell .code execution_count="112"}
``` {.python}
import numpy as np
A = np.array([[1.,2.,3.],[0.,1.,4.],[5.,6.,0.]])      #A n'est pas definie positive
B = np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])  #B est definie positive

Y=np.array([[45,-78,95]]).T                           #vecteur colonne choisie au hasard
```
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.3.2 La Matrice de test:`</u>`{=html}`</font>`{=html} {#iv32-la-matrice-de-test}
:::

::: {.cell .code execution_count="113" scrolled="true"}
``` {.python}
help(nps.gaussjordan)
```

::: {.output .stream .stdout}
    Help on function gaussjordan in module my_numpy.solve:

    gaussjordan(A, Y)
        gaussjordan(A,Y) takes a square matrix A and a column vector Y of the same length as A and
        if A is invertible it solves the system A@X=Y of linear equations using Gauss-Jordan method and returns the column vector solution X:
        otherwise it returns an error message.
:::
:::

::: {.cell .code execution_count="114" scrolled="true"}
``` {.python}
X=nps.gaussjordan(A,Y)
print("X =\n",X)
print("\n A@X=Y \n",A@X==Y,'\n')

print('---------------------------------------------------------------')
X=nps.gaussjordan(B,Y)
print("X =\n",X)
Y_=np.around(B@X,1)
print("\n B@X=Y \n",Y_==Y,'\n')

```

::: {.output .stream .stdout}
    X =
     [[-2009.]
     [ 1690.]
     [ -442.]]

     A@X=Y 
     [[ True]
     [ True]
     [ True]] 

    ---------------------------------------------------------------
    X =
     [[18.5]
     [-8. ]
     [43.5]]

     B@X=Y 
     [[ True]
     [ True]
     [ True]] 
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.3.3 LU:`</u>`{=html}`</font>`{=html} {#iv33-lu}
:::

::: {.cell .code execution_count="115" scrolled="false"}
``` {.python}
help(nps.LU)
```

::: {.output .stream .stdout}
    Help on function LU in module my_numpy.solve:

    LU(A, Y)
        LU(A,Y) takes a square matrix A and a column vector Y of the same length as A and
        
        if A is invertible it solves the system A@X=Y of linear equations using LU decomposition and returns the column vector solution X:
        otherwise it returns an error message.
:::
:::

::: {.cell .code execution_count="116"}
``` {.python}
X=nps.LU(A,Y)
print("X* =\n",X)
print("\nAX*=Y \n",A@X==Y)
print("-------------------------------------------------------------------------------");
X=nps.LU(B,Y)
print("X* =\n",X)
Y_=np.around(B@X,1)
print("\nBX*=Y\n",Y_==Y)
```

::: {.output .stream .stdout}
    X* =
     [[-2009.]
     [ 1690.]
     [ -442.]]

    AX*=Y 
     [[ True]
     [ True]
     [ True]]
    -------------------------------------------------------------------------------
    X* =
     [[18.5]
     [-8. ]
     [43.5]]

    BX*=Y
     [[ True]
     [ True]
     [ True]]
:::
:::

::: {.cell .markdown}
> > #### `<font color='blue'>`{=html}`<u>`{=html}IV.3.4 choleski:`</u>`{=html}`</font>`{=html} {#iv34-choleski}
:::

::: {.cell .code execution_count="117" scrolled="false"}
``` {.python}
help(nps.choleski)
```

::: {.output .stream .stdout}
    Help on function choleski in module my_numpy.solve:

    choleski(A, Y)
        choleski(A,Y) takes a square matrix A and a column vector Y of the same length as A and
        
        if A is positive definite it solves the system A@X=Y of linear equations using choleski decomposition and returns the column vector solution X:
        otherwise it returns an error message.
:::
:::

::: {.cell .code execution_count="118" scrolled="true"}
``` {.python}
X=nps.choleski(A,Y)
print("-------------------------------------------------------------------------------")
X=nps.choleski(B,Y)
print("X =\n",X)
Y_=np.around(B@X,1)
print("\nBX*=Y\n",Y_==Y)
```

::: {.output .stream .stdout}
    !! A must be positive definite !!
    -------------------------------------------------------------------------------
    X =
     [[18.5]
     [-8. ]
     [43.5]]

    BX*=Y
     [[ True]
     [ True]
     [ True]]
:::
:::

::: {.cell .markdown}
## Références:
:::

::: {.cell .markdown}
\[1\] M.Naoum, Lecture 3&4&5 - Optimization, non-linear programming :
One dimensional minimization methods.

\[2\] M.Naoum, Lecture 6 - Matrix Operations and Gaussian Elimination
for Solving Linear Systems.

\[3\] M.Naoum, Lecture 7 - Optimization, non-linear programming :
Multivariable Gradient Descent and Armijo\'s Condition.

\[4\] M.Essabri, Z.Mir & Y.Sedjari, Presentation - Stochastic Gradient
Descent and Backtracking Line Search.

\[5\] Christian, Scientific Blog - Visualizing the gradient descent
method:
<https://scipython.com/blog/visualizing-the-gradient-descent-method/>

\[6\] Three-Dimensional Plotting in Matplotlib :
<https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html>

\[7\] Data to Fish - How to Create a Horizontal Bar Chart using
Matplotlib : <https://datatofish.com/horizontal-bar-chart-matplotlib/>
:::
