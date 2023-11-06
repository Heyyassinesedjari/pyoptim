# <p align="center">Python-Library-for-Optimization</p>
<p align="center">
  <img src="https://img.shields.io/badge/conda-4.12.0-orange" width="10%" height="10%">
  <img src="https://img.shields.io/badge/Python-3.9.12-green" width="10%" height="10%">
  <img src="https://img.shields.io/badge/MIT_License-blue" width="8%" height="8%">
</p>

## A Python package integrating around ten unconstrained optimization algorithms, inclusive of 2D/3D visualizations for comparative analysis, and incorporated matrix operations.

This project, a component of the Numerical Analysis & Optimization course at [ENSIAS](https://fr.wikipedia.org/wiki/%C3%89cole_nationale_sup%C3%A9rieure_d%27informatique_et_d%27analyse_des_syst%C3%A8mes), [Mohammed V University](https://en.wikipedia.org/wiki/Mohammed_V_University) arose from Professor [M. Naoum](http://ensias.um5.ac.ma/professor/m-mohamed-naoum)'s suggestion to create a Python package encapsulating algorithms practiced during laboratory sessions. It focuses on about ten unconstrained optimization algorithms, offering 2D and 3D visualizations, enabling a performance comparison in computational efficiency and accuracy. Additionally, the project encompasses matrix operations such as inversion, decomposition, and solving linear systems, all integrated within the package containing the lab-derived algorithms.

<p align="center">
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/15197d66-cc2a-4149-8e21-db247ea31d25" width="50%" height="50%">
</p>
<p align="center">
  GIF Source: https://www.nsf.gov/news/mmg/mmg_disp.jsp?med_id=78950&from=](https://aria42.com/blog/2014/12/understanding-lbfgs
</p>

Every part of this project is sample code which shows how to do the following:
- Implementation of one-variable optimization algorithms, encompassing fixed_step, accelerated_step, exhaustive_search, dichotomous_search, interval_halving, Fibonacci, golden_section, Armijo_backward, and Armijo_forward.
- Incorporation of various one-variable optimization algorithms, including gradient descent, Gradient conjugate, Newton, quasi_Newton_dfp, stochastic gradient descent.
- Visualization of each iteration step for all algorithms in 2D, contour, and 3D formats.
- Conducting a comparative analysis focusing on their runtime and accuracy metrics.
- Implementation of matrix operations such as inversion (e.g., Gauss-Jordan), decomposition (e.g., LU, Choleski), and solutions for linear systems (e.g., Gauss-Jordan, LU, Choleski).
  

  ## Getting Started






  ## Documentation & Comparative Analysis

    ### Table of Contents
    - [Single Variable Optimization Algorithms](###single Variable Optimization Algorithms)
    - [Multivariable Optimization Algorithms](#Multivariable Optimization Algorithms)
    - [Matrix Inverse](#Matrix Inverse)
    - [Matrix Decomposition](#Matrix Decomposition)
    - [Solving Linear System](#Solving Linear Systems)
  
### Single Variable Optimization Algorithms

 #### Objectif Function
<p align="center">
  
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/fac9266e-0d8d-4685-80c0-5735e01cb541" width="50%" height="50%">
</p>

```python
def f(x):
    return x*(x-1.5)   # analytically, argmin(f) = 0.75
```

 #### Parameter Initialization
<p align="center">

  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/b0c201d3-8a75-42c9-97df-f989a124f8a8" width="50%" height="50%">
</p>

```python
xs=-10
xf=10
epsilon=1.e-2
```

#### importing libraries

```python
import my_scipy.onevar_optimize.minimize as soom
import my_plot.onevar._2D as po2
```



#### Fixed Step

Api call<br>
```python
print('x* =',soom.fixed_step(f,xs,epsilon))
po2.fixed_step(f,xs,epsilon)
```
```console
x* = 0.75
```
<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/5815e93a-3c54-4b70-a331-3ec13983e829" width="50%" height="50%">
</p>

#### Accelerated Step

Api call<br>
```python
print('x* =',soom.accelerated_step(f,xs,epsilon))
po2.accelerated_step(f,xs,epsilon)
```
```console
x* = 0.86
```


<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/237ae937-764b-4205-a50b-05ca82463c77" width="50%" height="50%">
</p>


#### Exhaustive Search
Api call<br>
```python
print('x* =',soom.exhaustive_search(f,xs,xf,epsilon))
po2.exhaustive_search(f,xs,xf,epsilon)
```
```console
x* = 0.75

```

<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/0431683d-b3d6-4cdf-b300-b1c7dc2fbfbe" width="50%" height="50%">
</p>


#### Dichotomous search 

Api call<br>
```python
mini_delta = 1.e-3
print('x* =',soom.dichotomous_search(f,xs,xf,epsilon,mini_delta))
po2.dichotomous_search(f,xs,xf,epsilon,mini_delta)
```
```console
x* = 0.7494742431640624

```


<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/98106569-afd9-431a-be84-b08563d291c8" width="50%" height="50%">
</p>

#### interval_halving

Api call<br>
```python
print('x* =',soom.interval_halving(f,xs,xf,epsilon))
po2.interval_halving(f,xs,xf,epsilon)
```
```console
x* = 0.75

```


<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/12c849d6-7660-4d0e-bfd2-ae475f17620a" width="50%" height="50%">
</p>

#### fibonacci

Api call<br>
```python
n=15
print('x* =',soom.fibonacci(f,xs,xf,n)) 
po2.fibonacci(f,xs,xf,n)
```
```console
x* = 0.76
```

<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/46ffc3d8-e215-4299-ba72-fd0e93d55b04" width="50%" height="50%">
</p>

#### golden_section

Api call<br>
```python
print('x* =',soom.golden_section(f,xs,xf,epsilon))
po2.golden_section(f,xs,xf,epsilon)
```
```console
x* = 0.75
```

<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/c264ebf3-6e08-4a95-811a-593f6866ffdf" width="50%" height="50%">
</p>

#### armijo_backward

Api call<br>
```python
ŋ=2
xs=100
print('x* =',soom.armijo_backward(f,xs,ŋ,epsilon))
po2.armijo_backward(f,xs,ŋ,epsilon)
```
```console
x* = 0.78
```

<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/5e09336e-a7a2-4927-9e16-f58b921d8152" width="50%" height="50%">
</p>

#### armijo_forward

Api call<br>
```python
xs=0.1
epsilon = 0.5
ŋ=2
print('x* =',soom.armijo_forward(f,xs,ŋ,epsilon))
po2.armijo_forward(f,xs,ŋ,epsilon)
```

```console
x* = 0.8
```



<p align="center">
  visulize each step<br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/422ae297-953e-4cb7-96f6-430547362417" width="50%" height="50%">
</p>


#### Comparative Analysis

Api call<br>
```python
po2.compare_all_time(f,0,2,1.e-2,1.e-3,10,2,0.1,100)
```
<p align="center">
  Runtime <br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/7faa0cff-9687-4dfb-b730-dbce7418172e" width="100%" height="100%">
</p>

Api call<br>
```python
po2.compare_all_precision(f,0,2,1.e-2,1.e-3,10,2,0.1,100)
```

<p align="center">
  Gap between true and computed argmin <br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/342df44c-8956-4d63-9d85-1850b82b138e" width="100%" height="100%">
</p>

### Multivariable Optimization Algorithms

#### import library
```python
import my_plot.multivar._3D as pm3
import my_plot.multivar.contour2D as pmc
```

<p align="center">
  Simple quadratic objectif function for isllustration purpose and Parameter Initialization<br>
</p>

```python
def h(x):
    return x[0] - x[1] + 2*(x[0]**2) + 2*x[1]*x[0] + x[1]**2
```

analitical solution

Parameter Initialization<br>
```python
X=[1000,897]
alpha=1.e-2
tol=1.e-2
```

#### gradient_descent

```python
pmc.gradient_descent(h,X,tol,alpha)
pm3.gradient_descent(h,X,tol,alpha)
```

```console
Y* =  [-1.01  1.51]
```

<p align="center">

<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/8a9885a7-681e-427d-a32d-384f33c653fd" width="100%" height="100%">
<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/d806e92f-256f-4b99-8fd7-4c911e0d9823" width="100%" height="100%">
</p>

#### gradient_conjugate

```python
pmc.gradient_conjugate(h,X,tol)
pm3.gradient_conjugate(h,X,tol)
```

```console
Y* =  [-107.38  172.18]
```

<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/4323c783-f219-489b-b89c-f500014ee53e" width="100%" height="100%">
<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/a696783b-8b6c-40d4-bef2-78c2e35a8283" width="100%" height="100%">


#### newton

```python
pmc.newton(h,X,tol)
pm3.newton(h,X,tol)
```

```console
Y* =  [-1.   1.5]
```

<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/f46c47b6-52b0-481e-aa28-4fb95bc92744" width="100%" height="100%">
<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/0d703a89-135e-48e5-b9e5-a4b797372f8f" width="100%" height="100%">

#### quasi_newton_dfp

```python
pmc.quasi_newton_dfp(h,X,tol)
pm3.quasi_newton_dfp(h,X,tol)
```
```console
Y* =  [-1.   1.5]
```

<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/f53ba118-1275-4d77-ab8b-9fbf0da9e5ca" width="100%" height="100%">
<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/0343a203-2634-4027-bdfc-9d1fe14dd479" width="100%" height="100%">

#### sgd

```python
pmc.sgd(h,X,tol,alpha)
pm3.sgd(h,X,tol,alpha)
```

```console
Y* =  [-1.01  1.52]
```


<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/2a7ba86a-0d6f-459b-9ea3-112e0aa6c39a" width="100%" height="100%">
<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/4b30a3da-6e1b-4eec-afe6-23cdec7427f6" width="100%" height="100%">

#### sgd_with_bls

```python
alpha = 100 #it must be high because of BLS
c = 2
pmc.sgd_with_bls(h,X,tol,alpha,c) 
pm3.sgd_with_bls(h,X,tol,alpha,c)
```

```console
Y* =  [-1.   1.5]
```

<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/de105b18-fddf-4794-8a0b-2b4b7192aeb9" width="100%" height="100%">
<img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/eefbe490-5531-492f-9070-8d7c7030ba8e" width="100%" height="100%">


#### Comparative Analysis

```python
pm3.compare_all_time(h,X,1.e-2,1.e-1,100,2)
```

<p align="center">
  Runtime <br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/a9593350-5745-4c0a-90aa-f257fd1b5bcd" width="100%" height="100%"> <br>
</p>

```python
pm3.compare_all_precision(h,X,1.e-2,1.e-1,100,2)
```

<p align="center">
  Gap between true and computed argmin <br>
  <img src="https://github.com/Heyyassinesedjari/Python-Library-for-Optimization/assets/94799575/b542577f-40e7-41db-b4f3-a8e6cdcca089" width="100%" height="100%">
</p>


### Matrix Inverse

#### Import Libraries

```python
import my_numpy.inverse as npi
```

#### Test Matrix

```python
A = np.array([[1.,2.,3.],[0.,1.,4.],[5.,6.,0.]])
```


#### gaussjordan
```python
A_1=npi.gaussjordan(A.copy())
I=A@A_1
I=np.around(I,1)
print('A_1 =\n\n',A_1)
print('\nA_1*A =\n\n',I)
```

```console
A_1 =

 [[-24.  18.   5.]
 [ 20. -15.  -4.]
 [ -5.   4.   1.]]

A_1*A =

 [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```


### Matrix Decomposition
#### Imports
```python
import my_numpy.decompose as npd
```

#### Test Matrix

```python
A = np.array([[1.,2.,3.],[0.,1.,4.],[5.,6.,0.]])      #A n'est pas definie positive 
B = np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])  #B est definie positive
Y=np.array([45,-78,95])                               #vecteur colonne choisie au hasard
```

#### LU


```python
L,U,P=npd.LU(A)
print("P =\n",P,"\n\nL =\n",L,"\n\nU =\n",U)
print("\n",A==P@L@U)
```


```console
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
``` 

##### Choleski

```python
L=npd.choleski(A)          # A is not positive definite
print(L)
print("--------------------------------------------------")
L=npd.choleski(B)          # B is positive definite 
print('L =\n',L,'\n')

C=np.around(L@(L.T),1)
print('B = L@(L.T) \n\n',B==C)
```

```console
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
```

### Solving Linear Systems

#### Imports
```python
import my_numpy.solve as nps
```


#### Test Matrix
```python
A = np.array([[1.,2.,3.],[0.,1.,4.],[5.,6.,0.]])      #A n'est pas definie positive 
B = np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])  #B est definie positive
Y=np.array([45,-78,95])                               #vecteur colonne choisie au hasard
```

#### gaussjordan

```python
X=nps.gaussjordan(A,Y)
print("X =\n",X)
print("\n A@X=Y \n",A@X==Y,'\n')

print('---------------------------------------------------------------')
X=nps.gaussjordan(B,Y)
print("X =\n",X)
Y_=np.around(B@X,1)
print("\n B@X=Y \n",Y_==Y,'\n')
```

```console
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

```

#### LU

```python
X=nps.LU(A,Y)
print("X* =\n",X)
print("\nAX*=Y \n",A@X==Y)
print("-------------------------------------------------------------------------------");
X=nps.LU(B,Y)
print("X* =\n",X)
Y_=np.around(B@X,1)
print("\nBX*=Y\n",Y_==Y)
```


```console
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
```

##### Choleski


```python
X=nps.choleski(A,Y)
print("-------------------------------------------------------------------------------")
X=nps.choleski(B,Y)
print("X =\n",X)
Y_=np.around(B@X,1)
print("\nBX*=Y\n",Y_==Y)
```


```console
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

```
